import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

LABELS = ['고열', '구토', '두통', '복통', '어지러움', '열상', '오심', '전신쇠약', '호흡곤란']
WORK = Path('/data1/workspaces/ddc/DCC_Final/Work/Qwen3_LoRA')
PRED_DIR = WORK / 'Predictions/qwen3_4b_qlora_v1'
VAL_FILE = WORK / 'Data/prefix_val_all.jsonl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='val_all')
    args = parser.parse_args()
    paths = sorted(PRED_DIR.glob(f'{args.tag}_rank*.jsonl'))
    if len(paths) != 8:
        raise RuntimeError(f'Expected 8 rank files; got {len(paths)}: {paths}')

    records = {}
    for path in paths:
        count = 0
        with path.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                key = (str(row['call_id']), int(row['step']))
                if key in records:
                    raise ValueError(f'Duplicate key: {key}')
                prob = np.asarray(row['probs'], dtype=np.float32)
                truth = np.asarray(row['label_vector'], dtype=np.int8)
                if prob.shape != (9,) or truth.shape != (9,) or not np.isfinite(prob).all():
                    raise ValueError(f'Invalid data at {key}')
                if ((prob < 0) | (prob > 1)).any() or not np.isin(truth, [0, 1]).all():
                    raise ValueError(f'Invalid prob/label at {key}')
                records[key] = (truth, prob)
                count += 1
        print(f'[LOAD] {path.name}: {count:,}', flush=True)

    # Compare with the original full validation manifest, catching missing/extra prefixes.
    expected = set()
    with VAL_FILE.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            key = (str(row['call_id']), int(row['step']))
            if key in expected:
                raise ValueError(f'Duplicate prefix in validation: {key}')
            expected.add(key)
    actual = set(records)
    if expected != actual:
        raise ValueError(f'Key mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')

    keys = sorted(records)
    y = np.stack([records[key][0] for key in keys])
    probs = np.stack([records[key][1] for key in keys])
    pred = (probs >= 0.5).astype(np.int8)
    p, r, f, sup = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
    print(f'\n[VALIDATED] Prefixes={len(keys):,}, calls={len(set(k[0] for k in keys)):,}')
    print(f'[THR=0.50] prefix macro F1={np.mean(f):.6f}, micro F1={f1_score(y, pred, average="micro", zero_division=0):.6f}')
    print('Class           Support  Precision    Recall        F1')
    for name, n, pv, rv, fv in zip(LABELS, sup, p, r, f):
        print(f'{name:<9} {n:>10}  {pv:>9.4f}  {rv:>8.4f}  {fv:>8.4f}')

    # Useful diagnostic; not a substitute for the exact official scoring script.
    call_stats = defaultdict(list)
    for idx, (call_id, _) in enumerate(keys):
        call_stats[call_id].append(idx)
    per_call_macro = np.mean([
        f1_score(y[idxs], pred[idxs], average='macro', zero_division=0)
        for idxs in call_stats.values()
    ])
    print(f'\n[DIAGNOSTIC] mean per-call prefix macro F1={per_call_macro:.6f}')

    out = PRED_DIR / f'{args.tag}_merged.jsonl'
    with out.open('w', encoding='utf-8') as f_out:
        for key in keys:
            truth, prob = records[key]
            f_out.write(json.dumps({
                'call_id': key[0], 'step': key[1],
                'label_vector': truth.tolist(), 'probs': [float(x) for x in prob],
            }, ensure_ascii=False) + '\n')
    print(f'[SAVED] {out}')
    print('[NOTE] These are diagnostic metrics, not a verified official DCC score.')


if __name__ == '__main__':
    main()
