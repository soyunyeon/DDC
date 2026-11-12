#!/usr/bin/env python3
"""Qwen v1 Mission 4-1 diagnostic. Read only original predictions/data; save analysis outputs."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/data1/workspaces/ddc/DCC_Final/Work/Qwen3_LoRA')
PRED = ROOT / 'Predictions/qwen3_4b_qlora_v1/val_all_merged.jsonl'
VAL = ROOT / 'Data/prefix_val_all.jsonl'
POLICY = ROOT / 'Predictions/qwen3_4b_qlora_v1/thresholds_official_experiment.json'
OUT = ROOT / 'Predictions/qwen3_4b_qlora_v1/error_analysis_18'
LABELS = ['고열','구토','두통','복통','어지러움','열상','오심','전신쇠약','호흡곤란']
BUCKETS = ('1-5','6-10','11-20','21+')
TARGET = {'오심','구토','두통','어지러움'}


def bucket(step):
    return '1-5' if step <= 5 else '6-10' if step <= 10 else '11-20' if step <= 20 else '21+'


def rows(path):
    with path.open('r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path.name}:{ln}: {e}') from e


def jaccard(gt, pd):
    union = gt | pd
    return len(gt & pd) / len(union) if union else 1.0


def thresholds(obj, step):
    if obj['type'] == 'two_stage':
        return obj['early_le10'] if step <= 10 else obj['late_ge11']
    if obj['type'] == 'global':
        return obj['thresholds']
    raise ValueError('Unknown threshold policy')


def main():
    for p in [PRED, VAL, POLICY]:
        if not p.is_file():
            raise FileNotFoundError(p)
    policy_file = json.loads(POLICY.read_text(encoding='utf-8'))
    if policy_file.get('labels') != LABELS:
        raise ValueError('label order mismatch')
    policy = policy_file['two_stage_official']
    OUT.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(Counter)
    label_stats = {b:{l:Counter() for l in LABELS} for b in ('ALL',)+BUCKETS}
    pair_counter = Counter()
    calls = defaultdict(lambda: {'sum':0.,'n':0,'base_sum':0.})
    examples = defaultdict(list)
    observed = set()
    n=0
    pred_iter = iter(rows(PRED))
    val_iter = iter(rows(VAL))
    # Prediction/validation file order may differ. Load prediction keys only; ~179k records.
    pred_by_key = {}
    for p in pred_iter:
        k=(str(p['call_id']),int(p['step']))
        if k in pred_by_key:
            raise ValueError(f'Duplicate prediction: {k}')
        pred_by_key[k] = p
    print(f'[LOAD] predictions={len(pred_by_key):,}', flush=True)
    for record in val_iter:
        key=(str(record['call_id']),int(record['step']))
        if key in observed:
            raise ValueError(f'Duplicate validation: {key}')
        observed.add(key)
        if key not in pred_by_key:
            raise ValueError(f'Missing prediction: {key}')
        p=pred_by_key[key]
        y=record['label_vector']
        if len(y)!=9 or len(p['probs'])!=9 or p['label_vector'] != y:
            raise ValueError(f'Label/prob mismatch: {key}')
        if any(v not in (0,1) for v in y):
            raise ValueError(f'Bad label vector: {key}')
        gt={i for i,v in enumerate(y) if v}
        prob=p['probs']
        t=thresholds(policy,key[1])
        pd={i for i,x in enumerate(prob) if x>=t[i]}
        baseline={i for i,x in enumerate(prob) if x>=.5}
        b=bucket(key[1]); fpos=pd-gt; fneg=gt-pd
        sc=jaccard(gt,pd); sc0=jaccard(gt,baseline)
        c=calls[key[0]];c['sum']+=sc;c['base_sum']+=sc0;c['n']+=1
        n+=1
        for group in ('ALL',b):
            s=stats[group];s['n']+=1;s['score_sum']+=sc;s['baseline_sum']+=sc0
            s['gt_count_sum']+=len(gt);s['pred_count_sum']+=len(pd)
            s['exact']+=int(gt==pd)
            s['over']+=int(len(pd)>len(gt));s['under']+=int(len(pd)<len(gt))
            s['equal_card_wrong']+=int(len(pd)==len(gt) and gt!=pd)
            s['fp_only']+=int(bool(fpos) and not fneg)
            s['fn_only']+=int(bool(fneg) and not fpos)
            s['both']+=int(bool(fpos) and bool(fneg))
            s['empty']+=int(not pd)
            for idx,lab in enumerate(LABELS):
                ls=label_stats[group][lab]
                ls['tp']+=int(idx in gt and idx in pd)
                ls['fp']+=int(idx not in gt and idx in pd)
                ls['fn']+=int(idx in gt and idx not in pd)
                ls['tn']+=int(idx not in gt and idx not in pd)
        for i in gt-pd:
            for j in pd-gt:
                pair_counter[(LABELS[i],LABELS[j])]+=1
        # Capture high-confidence error examples, no raw text emitted to terminal.
        if any(LABELS[i] in TARGET for i in (fpos|fneg)):
            error_count=len(fpos)+len(fneg)
            sample={
                'call_id':key[0], 'step':key[1], 'bucket':b,
                'gold':[LABELS[i] for i in sorted(gt)],
                'pred':[LABELS[i] for i in sorted(pd)],
                'false_positive':[LABELS[i] for i in sorted(fpos)],
                'false_negative':[LABELS[i] for i in sorted(fneg)],
                'score':round(sc,4),
                'probs':{LABELS[i]:round(float(prob[i]),4) for i in range(9)},
                'text':record.get('text',''),
            }
            examples[b].append((sc,-error_count,sample))
    if observed!=set(pred_by_key):
        raise ValueError(f'Extra predictions={len(set(pred_by_key)-observed)}')
    if n!=179169 or len(calls)!=5840:
        raise ValueError(f'Unexpected dataset size rows={n} calls={len(calls)}')
    summary={}
    print(f'[VALIDATED] rows={n:,} calls={len(calls):,}')
    for b in ('ALL',)+BUCKETS:
        s=stats[b];num=s['n']
        summary[b]={k:(v/num if k not in ('n','score_sum','baseline_sum','gt_count_sum','pred_count_sum') else v) for k,v in s.items()}
        print(f"[{b:5}] rows={num:6d} pooled_score={s['score_sum']/num:.4f} exact={s['exact']/num:.3f} over={s['over']/num:.3f} under={s['under']/num:.3f} both_errors={s['both']/num:.3f} avg_gold={s['gt_count_sum']/num:.2f} avg_pred={s['pred_count_sum']/num:.2f}")
    # Official score is equal-weighted over calls, not pooled prefixes.
    official=sum(c['sum']/c['n'] for c in calls.values())/len(calls)
    official0=sum(c['base_sum']/c['n'] for c in calls.values())/len(calls)
    print(f'[OFFICIAL] baseline_0.5={official0:.6f} two_stage_v1={official:.6f}')
    print('[LABEL ERRORS] ALL:')
    for lab in LABELS:
        c=label_stats['ALL'][lab];tp,fp,fn=c['tp'],c['fp'],c['fn']
        prec=tp/(tp+fp) if tp+fp else 0
        rec=tp/(tp+fn) if tp+fn else 0
        print(f'  {lab:5} tp={tp:6} fp={fp:6} fn={fn:6} precision={prec:.3f} recall={rec:.3f}')
    print('[CONFUSION] missed gold -> extra predicted (co-occurring errors, not exclusive):')
    for (a,b),cnt in pair_counter.most_common(15):
        print(f'  {a} -> {b}: {cnt}')
    with (OUT/'label_errors_by_bucket.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['bucket','symptom','TP','FP','FN','TN','precision','recall','f1'])
        for group in ('ALL',)+BUCKETS:
            for lab in LABELS:
                c=label_stats[group][lab];tp,fp,fn,tn=(c[x] for x in ('tp','fp','fn','tn'))
                pp=tp/(tp+fp) if tp+fp else 0;rr=tp/(tp+fn) if tp+fn else 0
                w.writerow([group,lab,tp,fp,fn,tn,round(pp,6),round(rr,6),round(2*pp*rr/(pp+rr),6) if pp+rr else 0])
    with (OUT/'call_scores.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['call_id','num_prefixes','official_0.5','official_two_stage'])
        for cid,c in sorted(calls.items()):
            w.writerow([cid,c['n'],round(c['base_sum']/c['n'],6),round(c['sum']/c['n'],6)])
    # Limit saved examples and keep them PRIVATE as transcripts may contain personally identifiable information.
    with (OUT/'error_examples_PRIVATE.jsonl').open('w',encoding='utf-8') as f:
        for b in BUCKETS:
            for _,_,rec in sorted(examples[b],key=lambda z:(z[0],z[1]))[:25]:
                f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    (OUT/'summary.json').write_text(json.dumps({'official_baseline':official0,'official_two_stage':official,'bucket_stats':summary,
        'label_errors':{g:{l:dict(v) for l,v in cs.items()} for g,cs in label_stats.items()},
        'coincident_errors_top':[[*k,v] for k,v in pair_counter.most_common(30)]},ensure_ascii=False,indent=2),encoding='utf-8')
    print('[SAVED]',OUT)
    print('[PRIVACY] Do not paste raw error_examples_PRIVATE.jsonl publicly; it may contain personal data.')
    print('[DONE] Read-only analysis of input files.')

if __name__=='__main__':
    main()
