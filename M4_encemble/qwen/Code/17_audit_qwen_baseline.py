#!/usr/bin/env python3
"""DCC M4-1: leakage-safe train-prior baseline + Qwen input/label audit. CPU only.
Does NOT train or modify existing files. Baseline policy is selected on TRAIN labels only.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path('/data1/workspaces/ddc/DCC_Final')
WORK = ROOT / 'Work/Qwen3_LoRA'
COMMON = ROOT / 'Work/Common'
NAMES = ['고열','구토','두통','복통','어지러움','열상','오심','전신쇠약','호흡곤란']


def read_rows(file):
    with file.open(encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_calls(file):
    d = {}
    for row in read_rows(file):
        cid = str(row['call_id'])
        assert cid not in d, f'duplicate call: {cid}'
        labels = np.asarray(row['label_vector'], dtype=np.uint8)
        assert labels.shape == (9,) and np.isin(labels, [0,1]).all() and labels.any()
        assert int(row['n_utterances']) == len(row['utterances'])
        d[cid] = row
    return d


def to_mask(labels):
    return sum((int(v) << k) for k, v in enumerate(labels))


def score_for_fixed_set(hist, mask):
    # For a constant prediction, per-step and per-call official score are identical.
    # Jaccard = |gold & pred| / |gold | pred|, with empty-empty=1.
    counts = np.arange(512, dtype=np.int32)
    intersection = np.fromiter(((int(i) & mask).bit_count() for i in counts), float)
    union = np.fromiter(((int(i) | mask).bit_count() for i in counts), float)
    scores = np.divide(intersection, union, out=np.ones(512), where=union > 0)
    return float((hist * scores).sum() / hist.sum())


def make_hist(calls):
    return np.bincount([to_mask(r['label_vector']) for r in calls.values()], minlength=512)


def main():
    train = load_calls(COMMON / 'mission4_train.jsonl')
    val = load_calls(COMMON / 'mission4_val.jsonl')
    assert len(train)==23360 and len(val)==5840
    assert not (set(train) & set(val)), 'TRAIN/VAL call overlap!'
    print(f'[SPLIT] TRAIN calls={len(train):,}, VAL calls={len(val):,}, overlap=0', flush=True)
    ht, hv = make_hist(train), make_hist(val)
    print('[PREVALENCE] symptom | train_count/rate | val_count/rate')
    for k, label in enumerate(NAMES):
        ct = sum(v for i,v in enumerate(ht) if i & (1<<k))
        cv = sum(v for i,v in enumerate(hv) if i & (1<<k))
        print(f'  {label:5s}: {ct:5d} / {ct/len(train):.3f} | {cv:5d} / {cv/len(val):.3f}')
    print('[LABEL-CARDINALITY] counts per call (TRAIN / VAL)')
    for x in range(1,10):
        a = sum(v for m,v in enumerate(ht) if m.bit_count()==x)
        b = sum(v for m,v in enumerate(hv) if m.bit_count()==x)
        if a or b: print(f'  {x}: {a} / {b}')

    fitted = sorted(((score_for_fixed_set(ht, m), m) for m in range(512)), reverse=True)
    best_train, best_mask = fitted[0]
    best_val = score_for_fixed_set(hv, best_mask)
    names = [name for k,name in enumerate(NAMES) if best_mask & (1<<k)]
    print(f'[TRAIN-ONLY PRIOR] chosen symptoms={names}')
    print(f'[TRAIN-ONLY PRIOR] official4-1 train={best_train:.6f} val={best_val:.6f}')
    print(f'[EMPTY-SET] official4-1 val={score_for_fixed_set(hv,0):.6f}')
    print('[REFERENCE] Qwen v1 official4-1@0.5=0.431520; Qwen v1 two-stage=0.476077')
    print('  Prior uses ONLY training label frequencies; it ignores conversation text and prefix step.')

    # Audit actual saved per-prefix inputs against common conversation source.
    rng = random.Random(42)
    val_prefix_file = WORK / 'Data/prefix_val_all.jsonl'
    v1_train_prefix_file = WORK / 'Data/prefix_train_8.jsonl'
    token_sample = []
    num_val = 0
    num_checked = 0
    num_truncated_text = 0
    prefixes_per_call = Counter()
    last_step = defaultdict(int)
    for r in read_rows(val_prefix_file):
        num_val += 1
        cid = str(r['call_id']); step = int(r['step'])
        assert cid in val and 1 <= step <= len(val[cid]['utterances'])
        assert step == last_step[cid]+1, f'Non-sequential VAL steps: {cid} {step}'
        last_step[cid] = step
        prefixes_per_call[cid] += 1
        assert r['label_vector'] == val[cid]['label_vector'], f'VAL label mismatch {cid}'
        if num_val <= 500 or rng.random() < 0.006:
            utt = val[cid]['utterances'][:step]
            expected = '\n'.join(f"{'신고자' if u['speaker']==1 else '119대원' if u['speaker']==0 else '화자미상'}: {u['text']}" for u in utt)
            assert expected == r['text'], f'VAL input mismatch {cid}/{step}'
            num_checked += 1
            if len(token_sample) < 1500: token_sample.append(r['text'])
    assert num_val == 179169 and len(prefixes_per_call) == len(val)
    assert all(last_step[cid] == len(row['utterances']) for cid,row in val.items())
    print(f'[VAL PREFIX AUDIT] rows={num_val:,} sampled_text_matches={num_checked:,}, all_steps_contiguous=YES, labels_match=YES')

    steps_by_call = defaultdict(set)
    for r in read_rows(v1_train_prefix_file):
        cid = str(r['call_id']); step=int(r['step'])
        assert cid in train and r['label_vector'] == train[cid]['label_vector']
        assert 1 <= step <= len(train[cid]['utterances'])
        assert step not in steps_by_call[cid], f'duplicate training prefix {cid}/{step}'
        steps_by_call[cid].add(step)
    assert len(steps_by_call)==len(train)
    print(f'[TRAIN PREFIX AUDIT] calls={len(steps_by_call):,}, rows={sum(map(len,steps_by_call.values())):,}, max_per_call={max(map(len,steps_by_call.values()))}')

    try:
        from transformers import AutoTokenizer
        model_path = WORK / 'Models/Qwen3-4B'
        tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        sizes = [len(ids) for ids in tok([x for x in token_sample], add_special_tokens=True, truncation=False)['input_ids']]
        a = np.asarray(sizes)
        print(f'[TOKEN SAMPLE] n={len(a)}, median={np.median(a):.0f}, p95={np.percentile(a,95):.0f}, max={a.max()}, >1024={int((a>1024).sum())} ({(a>1024).mean():.2%})')
        print('  NOTE: this is a sample, not the full token distribution; right truncation loses latest content above 1024 tokens.')
    except (ImportError, OSError) as e:
        print(f'[TOKEN SAMPLE SKIPPED] {type(e).__name__}: {e}')
    out = WORK / 'Predictions/qwen3_4b_qlora_v1/prior_baseline_audit.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'train_only_prior_symptoms':names,'train_only_prior_mask':best_mask,
                               'train_official_4_1':best_train,'val_official_4_1':best_val,
                               'n_val_prefixes':num_val,'n_calls_val':len(val)},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[SAVED] {out}')
    print('[DONE] read-only audit complete; no model training.')

if __name__=='__main__': main()
