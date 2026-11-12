import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score

ROOT = Path('/data1/workspaces/ddc/DCC_Final/Work/Qwen3_LoRA')
INFILE = ROOT / 'Predictions/qwen3_4b_qlora_v1/val_all_merged.jsonl'
OUTFILE = ROOT / 'Predictions/qwen3_4b_qlora_v1/thresholds_v1.json'
LABELS = ['고열','구토','두통','복통','어지러움','열상','오심','전신쇠약','호흡곤란']

ids, yy, pp = [], [], []
with INFILE.open(encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        ids.append(str(d['call_id']))
        yy.append(d['label_vector'])
        pp.append(d['probs'])
y = np.asarray(yy, dtype=np.int8)
p = np.asarray(pp, dtype=np.float32)
assert y.shape == p.shape == (179169, 9), (y.shape, p.shape)
assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()

# Split by call ID to prevent prefixes from the same call entering both sets.
train_mask = np.array([int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16) % 2 == 0 for cid in ids])
test_mask = ~train_mask
assert train_mask.any() and test_mask.any()
print('Tuning prefixes:', int(train_mask.sum()), 'Holdout prefixes:', int(test_mask.sum()))

def class_metrics(mask, thresholds):
    pred = p[mask] >= np.asarray(thresholds, dtype=np.float32)
    prec, rec, f1, sup = precision_recall_fscore_support(y[mask], pred, average=None, zero_division=0)
    return float(f1.mean()), f1, prec, rec, sup

# Threshold grid deliberately includes low thresholds for low-recall labels.
grid = np.arange(0.05, 0.951, 0.01)
chosen = []
for j in range(9):
    best_t, best_f1 = 0.5, -1.0
    yj = y[train_mask, j].astype(bool)
    pj = p[train_mask, j]
    positives = int(yj.sum())
    for t in grid:
        pred = pj >= t
        tp = int(np.count_nonzero(pred & yj))
        fp = int(np.count_nonzero(pred & ~yj))
        fn = positives - tp
        score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if score > best_f1 + 1e-12:
            best_t, best_f1 = float(round(t, 2)), float(score)
    chosen.append(best_t)

for label, mask in [('TUNE', train_mask), ('HOLDOUT', test_mask), ('ALL', np.ones(len(y), dtype=bool))]:
    base, bf, _, _, _ = class_metrics(mask, [0.5] * 9)
    tuned, tf, pr, re, su = class_metrics(mask, chosen)
    print(f'\n[{label}] macroF1 at 0.50={base:.6f}, optimized={tuned:.6f}, delta={tuned-base:+.6f}')
    print('Symptom          thr  support  P       R       F1base  F1new')
    for name, t, n, a, b, before, after in zip(LABELS, chosen, su, pr, re, bf, tf):
        print(f'{name:<12} {t:.2f} {n:>8}  {a:.4f}  {b:.4f}  {before:.4f}  {after:.4f}')

payload = {'labels': LABELS, 'thresholds': chosen,
           'selection': 'per-label prefix F1 on deterministic 50% call-level tuning split',
           'note': 'Held-out diagnostic only, not verified DCC official score'}
OUTFILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n[SAVED]', OUTFILE)
