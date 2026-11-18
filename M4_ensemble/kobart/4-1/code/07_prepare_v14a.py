import json
import random
from pathlib import Path
from collections import defaultdict, Counter

rng = random.Random(42)

src = Path("Work/Seq2Seq/train_prefix.jsonl")
val_src = Path("Work/Seq2Seq/val_prefix.jsonl")
out = Path("Work/V14A/train_prefix.jsonl")
stats_path = Path("Work/V14A/preprocess_stats.json")

assert src.exists(), f"Train 파일 없음: {src}"
assert val_src.exists(), f"Validation 파일 없음: {val_src}"
assert not out.exists(), f"기존 파일 보호: {out}"

with open("Work/Common/label_map.json", encoding="utf-8") as f:
    labels = json.load(f)["labels"]

ni = labels.index("오심")
vi = labels.index("구토")

rows = []
pools = {
    "vomit_only": defaultdict(list),
    "nausea_only": defaultdict(list),
    "both": defaultdict(list),
}

train_calls = set()

with src.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        y = r["target_vector"]

        assert len(y) == 9

        call_id = r["call_id"]
        train_calls.add(call_id)

        idx = len(rows)
        rows.append(r)

        nausea = int(y[ni])
        vomit = int(y[vi])

        if vomit and not nausea:
            pools["vomit_only"][call_id].append(idx)
        elif nausea and not vomit:
            pools["nausea_only"][call_id].append(idx)
        elif nausea and vomit:
            pools["both"][call_id].append(idx)

# Train / Validation 통화 분리 확인
val_calls = set()

with val_src.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        val_calls.add(r["call_id"])

overlap = train_calls & val_calls
assert not overlap, f"Train/Val 통화 중복: {len(overlap)}"

print("원본 Train Prefix:", f"{len(rows):,}")
print("Train 통화:", f"{len(train_calls):,}")
print("Validation 통화:", f"{len(val_calls):,}")
print("통화 중복:", len(overlap))

# 기존 Train을 유지하면서 그룹별 5% 보강
extra_each = round(len(rows) * 0.05)
extra_indices = []
summary = {}

for name, by_call in pools.items():
    call_ids = list(by_call.keys())
    assert call_ids, f"샘플 부족: {name}"

    rng.shuffle(call_ids)

    selected = []

    # 특정 긴 통화에 보강 샘플이 집중되지 않도록
    # 통화를 순환 선택한 뒤 Prefix를 무작위 선택
    for k in range(extra_each):
        cid = call_ids[k % len(call_ids)]
        selected.append(rng.choice(by_call[cid]))

    extra_indices.extend(selected)

    summary[name] = {
        "available_calls": len(call_ids),
        "available_prefixes": sum(map(len, by_call.values())),
        "extra_samples": len(selected)
    }

    print(
        f"{name}: 통화 {len(call_ids):,}개 / "
        f"추가 {len(selected):,}개"
    )

# 전체 기존 Train + 보강 Train
indices = list(range(len(rows))) + extra_indices
rng.shuffle(indices)

out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    for i in indices:
        f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")

assert len(indices) == len(rows) + len(extra_indices)

summary["original_prefixes"] = len(rows)
summary["extra_prefixes"] = len(extra_indices)
summary["total_prefixes"] = len(indices)
summary["train_calls"] = len(train_calls)
summary["validation_overlap"] = len(overlap)
summary["seed"] = 42

stats_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n===== V14-A 전처리 완료 =====")
print("원본:", f"{len(rows):,}")
print("추가:", f"{len(extra_indices):,}")
print("전체:", f"{len(indices):,}")
print("저장:", out)
