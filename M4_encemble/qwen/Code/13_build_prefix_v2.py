#!/usr/bin/env python3
"""DCC M4 Qwen3: controlled prefix-sampling comparison; data-only, CPU.

v1 selection: [1,2,3,5,8,round(.4*n),round(.7*n),n]
v2 selection: [1,2,3,4,5,8,round(.7*n),n]
Both have <=8 unique samples per call, so compare at approximately equal budget.
The entire known label remains constant as in the official prefix task.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/data1/workspaces/ddc/DCC_Final')
SOURCE = ROOT / 'Work/Common/mission4_train.jsonl'
OLD = ROOT / 'Work/Qwen3_LoRA/Data/prefix_train_8.jsonl'
OUTPUT = ROOT / 'Work/Qwen3_LoRA/Data/prefix_train_v2_early8.jsonl'


def v1_steps(n):
    return sorted({min(n, x) for x in [1, 2, 3, 5, 8, max(1, round(n*.4)), max(1, round(n*.7)), n]})


def v2_steps(n):
    return sorted({min(n, x) for x in [1, 2, 3, 4, 5, 8, max(1, round(n*.7)), n]})


def who(sp):
    return '신고자' if sp == 1 else '119대원' if sp == 0 else '화자미상'


def main():
    assert SOURCE.is_file() and OLD.is_file(), 'Missing common train or v1 data'
    if OUTPUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUTPUT}')
    cnt_v1, cnt_v2, hist_v1, hist_v2 = Counter(), Counter(), Counter(), Counter()
    n_calls, n_v1, n_v2, different = 0, 0, 0, 0
    ids = set()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(encoding='utf-8') as inp, OUTPUT.open('x', encoding='utf-8') as out:
        for raw in inp:
            if not raw.strip():
                continue
            r = json.loads(raw)
            assert r['split'] == 'train'
            cid = r['call_id']
            assert cid not in ids, f'Duplicate call: {cid}'
            ids.add(cid)
            utt = r['utterances']
            n = len(utt)
            assert n > 0 and n == r['n_utterances']
            label = r['label_vector']
            assert len(label) == 9 and all(x in (0, 1) for x in label)
            assert [u['index'] for u in utt] == list(range(n))
            a, b = v1_steps(n), v2_steps(n)
            assert 1 <= len(a) <= 8 and 1 <= len(b) <= 8
            n_v1 += len(a)
            n_v2 += len(b)
            different += (a != b)
            hist_v1[len(a)] += 1
            hist_v2[len(b)] += 1
            cnt_v1.update(a)
            cnt_v2.update(b)
            chunks = []
            target = set(b)
            for i, u in enumerate(utt, 1):
                chunks.append(f"{who(u['speaker'])}: {u['text']}")
                if i in target:
                    rec = {'call_id': cid, 'label_file': r['label_file'], 'step': i,
                           'text': '\n'.join(chunks), 'label_vector': label,
                           'symptom': r['symptom']}
                    out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_calls += 1
            if n_calls % 5000 == 0:
                print(f'[PROGRESS] calls={n_calls:,}', flush=True)
    assert n_calls == 23360, f'Unexpected call count {n_calls}'
    old_count = sum(bool(line.strip()) for line in OLD.open(encoding='utf-8'))
    assert old_count == n_v1, f'Old dataset mismatch: {old_count} vs {n_v1}'
    print(f'[DONE] calls={n_calls:,} original={n_v1:,} v2={n_v2:,} changed_calls={different:,}')
    print(f'[HIST V1] {dict(sorted(hist_v1.items()))}')
    print(f'[HIST V2] {dict(sorted(hist_v2.items()))}')
    for name, lo, hi in [('1-3', 1, 3), ('4-5', 4, 5), ('6-10', 6, 10), ('11-20', 11, 20), ('21+', 21, 10**9)]:
        x = sum(v for k,v in cnt_v1.items() if lo <= k <= hi)
        y = sum(v for k,v in cnt_v2.items() if lo <= k <= hi)
        print(f'[STEPS {name}] v1={x:,} v2={y:,} delta={y-x:+,}')
    print(f'[SAVED] {OUTPUT}')
    print('[NOTE] Validation and original train data were not modified.')


if __name__ == '__main__':
    main()
