#!/usr/bin/env python3
"""Mission 4-1 official-score threshold experiments: Qwen3 standalone, CPU only.
Deterministic per-call TUNE/HOLDOUT split matches scripts 10 and 11.
HOLDOUT is evaluated only, never used for optimization.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path('/data1/workspaces/ddc/DCC_Final/Work/Qwen3_LoRA')
INFILE = ROOT / 'Predictions/qwen3_4b_qlora_v1/val_all_merged.jsonl'
OUTDIR = ROOT / 'Predictions/qwen3_4b_qlora_v1'
LABELS = ['고열','구토','두통','복통','어지러움','열상','오심','전신쇠약','호흡곤란']
GRID = np.round(np.arange(0.10, 0.901, 0.05), 2)


def group(cid):
    return int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16) % 2 == 0


def load():
    ids, steps, y, p = [], [], [], []
    seen = set()
    with INFILE.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            cid, step = str(row['call_id']), int(row['step'])
            key = (cid, step)
            if key in seen:
                raise ValueError(f'Duplicate prefix: {key}')
            seen.add(key)
            ids.append(cid)
            steps.append(step)
            y.append(row['label_vector'])
            p.append(row['probs'])
    y = np.asarray(y, dtype=np.bool_)
    p = np.asarray(p, dtype=np.float32)
    steps = np.asarray(steps, dtype=np.int32)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 9:
        raise ValueError(f'Invalid shapes: {y.shape}, {p.shape}')
    if not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError('Invalid probabilities')
    if len(ids) != 179169 or len(set(ids)) != 5840:
        raise ValueError(f'Unexpected rows/calls: {len(ids)}, {len(set(ids))}')
    cid_idx = {cid: idx for idx, cid in enumerate(sorted(set(ids)))}
    call_index = np.asarray([cid_idx[cid] for cid in ids], dtype=np.int32)
    counts = np.bincount(call_index)
    if np.any(counts < 1):
        raise ValueError('Empty call')
    weights = 1.0 / counts[call_index].astype(np.float64)
    tune = np.asarray([group(cid) for cid in ids], dtype=np.bool_)
    # Account for missing or duplicate steps within each call.
    if np.any(np.bincount(call_index, weights=steps) != counts * (counts + 1) / 2):
        raise ValueError('Non-contiguous step indexes (sum mismatch)')
    return y, p, steps, weights, tune


def moment(y, pred):
    tp = np.logical_and(y, pred).sum(axis=1, dtype=np.int16)
    un = np.logical_or(y, pred).sum(axis=1, dtype=np.int16)
    return np.divide(tp, un, out=np.ones(len(y), dtype=np.float64), where=un > 0)


def official_score(y, p, step, weight, split, policy):
    pred = np.empty_like(y)
    if policy['type'] == 'global':
        pred[:] = p >= np.asarray(policy['thresholds'], dtype=np.float32)
    elif policy['type'] == 'two_stage':
        early = step <= 10
        pred[early] = p[early] >= np.asarray(policy['early_le10'], dtype=np.float32)
        pred[~early] = p[~early] >= np.asarray(policy['late_ge11'], dtype=np.float32)
    else:
        raise ValueError(policy['type'])
    vals = moment(y[split], pred[split])
    w = weight[split]
    return float(np.dot(vals, w) / w.sum())


def fit_thresholds(y, p, weights, initial):
    """Coordinate ascent of sum(moment Jaccard * inverse call length).
    These are scores from selected rows, and unselected rows are held constant.
    """
    thresholds = np.asarray(initial, dtype=np.float32).copy()
    best_score = -1.0
    for cycle in range(3):
        updates = 0
        for j in range(9):
            pred = p >= thresholds
            tp = np.logical_and(y, pred).sum(axis=1, dtype=np.int16)
            union = np.logical_or(y, pred).sum(axis=1, dtype=np.int16)
            old = pred[:, j]
            old_tp = old & y[:, j]
            old_un = old | y[:, j]
            jbest = -1.0
            chosen = float(thresholds[j])
            for t in GRID:
                new = p[:, j] >= np.float32(t)
                new_tp = tp - old_tp + (new & y[:, j])
                new_un = union - old_un + (new | y[:, j])
                moment_scores = np.divide(new_tp, new_un,
                                          out=np.ones(len(y), dtype=np.float64),where=new_un > 0)
                score = float(np.dot(moment_scores, weights))
                # Ties favor closeness to default .50 to avoid unnecessary complexity.
                if score > jbest + 1e-9 or (abs(score-jbest) <= 1e-9 and abs(float(t)-.5)<abs(chosen-.5)):
                    jbest, chosen = score, float(t)
            if abs(chosen - float(thresholds[j])) > 1e-5:
                updates += 1
                thresholds[j] = chosen
        best_score = float(np.dot(moment(y, p >= thresholds), weights))
        print(f'    cycle {cycle+1}: weighted tune objective={best_score:.3f}, changes={updates}', flush=True)
        if updates == 0:
            break
    return [round(float(v),2) for v in thresholds]


def main():
    y,p,step,weight,tune=load()
    holdout=~tune
    # Each call has total weight 1, so score equals weighted sum / number of calls.
    n_tune = int(round(weight[tune].sum()))
    n_holdout = int(round(weight[holdout].sum()))
    if (n_tune,n_holdout)!=(2920,2920):
        raise ValueError(f'Unexpected split: {n_tune}, {n_holdout}')
    print(f'[DATA] rows={len(y):,} calls=5840 TUNE={n_tune} HOLDOUT={n_holdout}',flush=True)

    def report(name, policy):
        print(f'\n[{name}]',flush=True)
        for split_name,mask,calls in [('TUNE',tune,n_tune),('HOLDOUT',holdout,n_holdout),('ALL',np.ones(len(y),dtype=bool),5840)]:
            val=official_score(y,p,step,weight,mask,policy)
            # official_score computes sum(weights*m) / sum(weights), equal to call average
            print(f'  {split_name:<7} official_4_1={val:.6f}',flush=True)
        print('  policy=',json.dumps(policy,ensure_ascii=False),flush=True)

    base={'type':'global','thresholds':[0.5]*9}
    report('BASELINE_0.5',base)
    print('\n[FIT] global 9 thresholds (TUNE only)',flush=True)
    g=fit_thresholds(y[tune],p[tune],weight[tune],[0.5]*9)
    global_policy={'type':'global','thresholds':g}
    report('GLOBAL_OFFICIAL',global_policy)

    print('\n[FIT] two-stage 9+9 thresholds, early=1..10, late=11+ (TUNE only)',flush=True)
    early = step <= 10
    a=fit_thresholds(y[tune & early],p[tune & early],weight[tune & early],g)
    b=fit_thresholds(y[tune & ~early],p[tune & ~early],weight[tune & ~early],g)
    stage_policy={'type':'two_stage','early_le10':a,'late_ge11':b}
    report('TWO_STAGE_OFFICIAL',stage_policy)

    payload={'labels':LABELS,'objective':'published Mission 4-1 per-moment Jaccard, per-call mean',
             'split':'sha256(call_id) first eight hex digits mod 2; 0=TUNE 1=HOLDOUT',
             'baseline':base,'global_official':global_policy,'two_stage_official':stage_policy,
             'warning':'Select on TUNE only. HOLDOUT is a one-time diagnostic, not official test.'}
    path=OUTDIR/'thresholds_official_experiment.json'
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'\n[SAVED] {path}\n[DONE] No model or original predictions changed.',flush=True)

if __name__=='__main__':
    main()
