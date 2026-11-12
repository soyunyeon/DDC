#!/usr/bin/env python3
"""Qwen3 standalone diagnostic: exact published M4-1 moment_score, per-call aggregation.
No GPU. Does not modify existing predictions or thresholds.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

WORK = Path('/data1/workspaces/ddc/DCC_Final/Work/Qwen3_LoRA')
PRED = WORK / 'Predictions/qwen3_4b_qlora_v1'
DATA = WORK / 'Data/prefix_val_all.jsonl'
LABELS = ['고열','구토','두통','복통','어지러움','열상','오심','전신쇠약','호흡곤란']
BUCKETS = [('1-5',1,5), ('6-10',6,10), ('11-20',11,20), ('21+',21,10**9)]


def group(cid):
    return 'TUNE' if int(hashlib.sha256(cid.encode()).hexdigest()[:8],16) % 2 == 0 else 'HOLDOUT'


def main():
    source = PRED / 'val_all_merged.jsonl'
    conf = json.loads((PRED / 'thresholds_v1.json').read_text(encoding='utf-8'))
    if conf['labels'] != LABELS:
        raise ValueError('Threshold label ordering mismatch')
    thresholds = np.asarray(conf['thresholds'], dtype=np.float32)
    if thresholds.shape != (9,) or not np.all((thresholds > 0) & (thresholds < 1)):
        raise ValueError('Invalid thresholds')

    keys, truth, probs = [], [], []
    seen = set()
    with source.open(encoding='utf-8') as f:
        for ln, line in enumerate(f,1):
            r = json.loads(line)
            k = (str(r['call_id']),int(r['step']))
            if k in seen: raise ValueError(f'Duplicate key: {k}')
            seen.add(k)
            keys.append(k)
            truth.append(r['label_vector'])
            probs.append(r['probs'])
    y = np.asarray(truth,dtype=np.bool_)
    p = np.asarray(probs,dtype=np.float32)
    if y.shape != p.shape or y.shape[1] != 9 or not np.isfinite(p).all():
        raise ValueError('Invalid rows')
    if len(keys) != 179169:
        raise ValueError(f'Unexpected prefix count: {len(keys)}')
    calls = defaultdict(list)
    for i,(cid,step) in enumerate(keys): calls[cid].append(i)
    if len(calls) != 5840: raise ValueError(f'Unexpected calls: {len(calls)}')
    for cid,inds in calls.items():
        steps = sorted(keys[i][1] for i in inds)
        if steps != list(range(1,len(steps)+1)):
            raise ValueError(f'Noncontiguous prefix steps in {cid}')
        if not (y[inds] == y[inds[0]]).all():
            raise ValueError(f'Inconsistent gold labels in {cid}')

    pred_base = p >= 0.5
    pred_v1 = p >= thresholds

    # Exactly matches official score.py's moment_score for nonempty gold:
    # TP / (TP + FP + FN), TN excluded. Then mean over moments per call, then calls.
    def moment_scores(pred):
        intersection = np.logical_and(y,pred).sum(axis=1)
        union = np.logical_or(y,pred).sum(axis=1)
        return np.divide(intersection,union,out=np.ones(len(keys),dtype=np.float64),where=(union != 0))

    m0, m1 = moment_scores(pred_base), moment_scores(pred_v1)
    splits = {'ALL':list(calls),
              'TUNE':[cid for cid in calls if group(cid)=='TUNE'],
              'HOLDOUT':[cid for cid in calls if group(cid)=='HOLDOUT']}
    print('[OFFICIAL M4-1 FORMULA] per-step TP/(TP+FP+FN), then per-call mean, then call mean')
    print(f'[CHECK] rows={len(keys):,}, calls={len(calls):,}, threshold={thresholds.tolist()}')
    for label, cids in splits.items():
        a = np.asarray([m0[calls[cid]].mean() for cid in cids])
        b = np.asarray([m1[calls[cid]].mean() for cid in cids])
        print(f'[{label}] calls={len(cids):,}  official@0.5={a.mean():.6f}  '
              f'official@v1={b.mean():.6f}  delta={b.mean()-a.mean():+.6f}')

    print('\n[STEP BUCKETS] per-call within-bucket average, calls equally weighted')
    for name,lo,hi in BUCKETS:
        for label,cids in [('ALL',splits['ALL']),('HOLDOUT',splits['HOLDOUT'])]:
            pairs = []
            for cid in cids:
                idx = [i for i in calls[cid] if lo <= keys[i][1] <= hi]
                if idx: pairs.append((m0[idx].mean(),m1[idx].mean()))
            x = np.asarray(pairs)
            print(f'[{name:>5} {label:>7}] calls={len(pairs):>4} '
                  f'base={x[:,0].mean():.6f} v1={x[:,1].mean():.6f} '
                  f'delta={(x[:,1]-x[:,0]).mean():+.6f}')

    print('\n[SYMPTOM F1 PER BUCKET] prefix pooled, not official score')
    steps = np.asarray([s for _,s in keys],dtype=np.int32)
    for name,lo,hi in BUCKETS:
        mask=(steps>=lo)&(steps<=hi)
        _,_,f0,_=precision_recall_fscore_support(y[mask],pred_base[mask],average=None,zero_division=0)
        _,_,f1,_=precision_recall_fscore_support(y[mask],pred_v1[mask],average=None,zero_division=0)
        print(f'{name}: n={int(mask.sum())} macro_base={f0.mean():.4f} macro_v1={f1.mean():.4f}')
        print('  '+' | '.join(f'{c} {a:.3f}->{b:.3f}' for c,a,b in zip(LABELS,f0,f1)))

    # Save per-call and case indexes. Text excerpts are resolved separately below.
    PRED.mkdir(parents=True,exist_ok=True)
    out = PRED / 'official_m41_per_call_v1.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f)
        w.writerow(['call_id','split','num_steps','score_0_5','score_threshold_v1','delta'])
        for cid,inds in sorted(calls.items()):
            a=float(m0[inds].mean());b=float(m1[inds].mean())
            w.writerow([cid,group(cid),len(inds),f'{a:.7f}',f'{b:.7f}',f'{b-a:.7f}'])
    print('[SAVED]',out)

    # Select top difficult moments for three target symptoms on HOLDOUT only.
    wanted=['오심','구토','두통']
    case_keys=set()
    cases=[]
    for symptom in wanted:
        j=LABELS.index(symptom)
        for kind,mask in [('FN',y[:,j]&~pred_v1[:,j]),('FP',~y[:,j]&pred_v1[:,j])]:
            # Put early moments first; then prefer stronger model confidence for FN/FP.
            cand=[i for i in np.flatnonzero(mask) if group(keys[i][0])=='HOLDOUT']
            cand.sort(key=lambda i:(keys[i][1], float(p[i,j]) if kind=='FN' else -float(p[i,j])))
            unique_calls=set()
            for i in cand:
                if keys[i][0] in unique_calls:continue
                unique_calls.add(keys[i][0]);case_keys.add(keys[i])
                cases.append({'call_id':keys[i][0],'step':keys[i][1],
                              'symptom':symptom,'error':kind,
                              'probability':round(float(p[i,j]),5),
                              'gold':int(y[i,j]),'predicted':int(pred_v1[i,j])})
                if len(unique_calls)>=12:break

    # Retrieve only matching prefixes to avoid exposing irrelevant call transcripts.
    context={}
    if DATA.exists():
        with DATA.open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                k=(str(r['call_id']),int(r['step']))
                if k in case_keys:
                    context[k]=r.get('text','')[-700:]
    for r in cases:r['prefix_excerpt']=context.get((r['call_id'],r['step']),'')
    case_file=PRED/'official_m41_error_cases_v1.jsonl'
    with case_file.open('w',encoding='utf-8') as f:
        for r in cases:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'[SAVED] {case_file} ({len(cases)} examples; keep private: may contain personal information)')
    print('[DONE] Analysis complete. No model retraining performed.')

if __name__=='__main__': main()
