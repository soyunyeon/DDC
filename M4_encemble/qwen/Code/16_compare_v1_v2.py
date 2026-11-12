#!/usr/bin/env python3
"""Evaluate Qwen3 v1/v2 on exactly the same DCC M4-1 validation prefixes.
Read only: rank outputs, v1 merged, saved v1 policies. CPU only.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK = Path('/data1/workspaces/ddc/DCC_Final/Work/Qwen3_LoRA')
ROOT = WORK / 'Predictions'
V1 = ROOT / 'qwen3_4b_qlora_v1'
V2 = ROOT / 'qwen3_4b_qlora_v2_early'
VAL = WORK / 'Data/prefix_val_all.jsonl'
LABELS = ['고열','구토','두통','복통','어지러움','열상','오심','전신쇠약','호흡곤란']
BUCKETS = [('1-5',1,5),('6-10',6,10),('11-20',11,20),('21+',21,10**8)]


def load_records(paths, name):
    out = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        cnt = 0
        with path.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                key = (str(row['call_id']), int(row['step']))
                if key in out:
                    raise RuntimeError(f'{name}: duplicate {key}')
                y = np.asarray(row['label_vector'],dtype=np.int8)
                p = np.asarray(row['probs'],dtype=np.float32)
                if y.shape != (9,) or p.shape != (9,) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
                    raise ValueError(f'{name}: invalid value {key}')
                out[key]=(y,p)
                cnt += 1
        print(f'[LOAD] {name} {path.name}: {cnt:,}',flush=True)
    return out


def moment(y, predicted):
    intersection = np.logical_and(y,predicted).sum(1)
    union = np.logical_or(y,predicted).sum(1)
    return np.divide(intersection,union,out=np.ones(len(y),dtype=np.float64),where=union>0)


def thresholds(policy, steps):
    if policy['type']=='global':
        t=np.asarray(policy['thresholds'],dtype=np.float32)
        if t.shape!=(9,): raise ValueError('Invalid global policy')
        return t
    if policy['type']=='two_stage':
        e=np.asarray(policy['early_le10'],dtype=np.float32)
        l=np.asarray(policy['late_ge11'],dtype=np.float32)
        if e.shape!=(9,) or l.shape!=(9,): raise ValueError('Invalid two-stage policy')
        return np.where((steps<=10)[:,None],e,l)
    raise ValueError('Unknown policy')


def callmeans(values, inds, eligible):
    return np.asarray([values[inds[c]].mean() for c in eligible],dtype=float)


def main():
    a=load_records([V1/'val_all_merged.jsonl'],'v1')
    b=load_records([V2/f'val_all_rank{r}.jsonl' for r in range(8)],'v2')
    expected=set()
    for line in VAL.open(encoding='utf-8'):
        r=json.loads(line); key=(str(r['call_id']),int(r['step']))
        if key in expected: raise ValueError(f'Validation duplicate: {key}')
        expected.add(key)
    if len(expected)!=179169: raise ValueError(f'Unexpected validation rows: {len(expected)}')
    if set(a)!=expected or set(b)!=expected:
        raise ValueError(f'Key mismatch: v1 missing={len(expected-set(a))}, v2 missing={len(expected-set(b))}, extras v1={len(set(a)-expected)}, v2={len(set(b)-expected)}')
    keys=sorted(expected)
    y=np.stack([a[k][0] for k in keys]).astype(bool)
    if any(not np.array_equal(a[k][0],b[k][0]) for k in keys):
        raise ValueError('v1/v2 gold label mismatch')
    p1=np.stack([a[k][1] for k in keys]); p2=np.stack([b[k][1] for k in keys]); steps=np.array([k[1] for k in keys]);
    inds=defaultdict(list)
    for i,(cid,step) in enumerate(keys):inds[cid].append(i)
    if len(inds)!=5840:raise ValueError('Unexpected call count')
    for cid,ii in inds.items():
        if steps[ii].tolist()!=list(range(1,len(ii)+1)):
            raise ValueError(f'Invalid steps: {cid}')
        if not (y[ii]==y[ii[0]]).all():raise ValueError(f'Inconsistent gold labels {cid}')
    calls=sorted(inds)
    tune=[c for c in calls if int(hashlib.sha256(c.encode()).hexdigest()[:8],16)%2==0]
    hold=[c for c in calls if c not in set(tune)]
    if len(tune)!=2920 or len(hold)!=2920:raise ValueError('Unexpected split')
    print(f'\n[VALIDATED] prefixes={len(keys):,} calls={len(calls):,} tune={len(tune)} holdout={len(hold)}')
    policies=json.loads((V1/'thresholds_official_experiment.json').read_text(encoding='utf-8'))
    if policies.get('labels')!=LABELS:raise ValueError('Policy class order differs')
    ps=[('baseline_0.5', {'type':'global','thresholds':[.5]*9}),('v1_global_policy',policies['global_official']),('v1_two_stage_policy',policies['two_stage_official'])]
    for name,policy in ps:
        t=thresholds(policy,steps)
        m1=moment(y,p1>=t); m2=moment(y,p2>=t)
        print(f'\n[POLICY] {name} (same thresholds applied to both models)')
        for label,cc in [('TUNE',tune),('HOLDOUT',hold),('ALL',calls)]:
            s1=callmeans(m1,inds,cc); s2=callmeans(m2,inds,cc)
            print(f'[{label:7}] v1={s1.mean():.6f} v2={s2.mean():.6f} delta={s2.mean()-s1.mean():+.6f}')
        print('[STEP BUCKETS] per-call within-bucket average; ALL')
        for bn,lo,hi in BUCKETS:
            cc=[c for c in calls if any(lo<=steps[i]<=hi for i in inds[c])]
            v1=np.mean([m1[[i for i in inds[c] if lo<=steps[i]<=hi]].mean() for c in cc]);
            v2=np.mean([m2[[i for i in inds[c] if lo<=steps[i]<=hi]].mean() for c in cc]);
            print(f'  {bn:5} calls={len(cc):4} v1={v1:.6f} v2={v2:.6f} delta={v2-v1:+.6f}')
    print('\n[DONE] No files changed; v2 thresholds not tuned yet.')

if __name__=='__main__':main()
