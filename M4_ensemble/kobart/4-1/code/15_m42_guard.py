import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("Results/M42_Learned")
ROOT.mkdir(parents=True, exist_ok=True)

PRED = Path("Results/V13A/early10_val.csv")
PROB = Path(
    "Results/M4_1/"
    "t5_t5_m41_v13a_targeted_3000_val_probs.jsonl"
)

LABELS = [
    "고열", "구토", "두통", "복통",
    "어지러움", "열상", "오심",
    "전신쇠약", "호흡곤란"
]
IDX = {x: i for i, x in enumerate(LABELS)}

def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {
            r["label_file"]: r
            for r in csv.DictReader(f)
        }

base = load_csv(ROOT / "baseline.csv")
learned = load_csv(ROOT / "learned035.csv")

assert base.keys() == learned.keys()
assert len(base) == 1431

# 시점별 증상 예측
preds = defaultdict(dict)

with PRED.open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        lf = r["label_file"]
        if lf not in base:
            continue
        t = int(r["step"])

        preds[lf][t] = frozenset(
            x.strip()
            for x in r["symptom"].split(";")
            if x.strip()
        )

# 시점별 T5 확률
probs = defaultdict(dict)

with PROB.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        lf = r["label_file"]
        if lf not in base:
            continue

        probs[lf][int(r["step"])] = np.asarray(
            r["probs"], dtype=np.float32
        )

# 시점 연속성 검사
for lf in base:
    steps = set(preds[lf])
    assert steps == set(probs[lf])
    assert steps == set(range(1, len(steps) + 1))

def safe_to_stop(lf, t, window, delta_limit, min_prob):
    if t < max(6, window):
        return False

    now = preds[lf][t]

    if not now:
        return False

    # 최근 N회 동일한 증상 예측
    for k in range(1, window):
        if preds[lf][t-k] != now:
            return False

    # 최근 N회 확률의 최대 변화
    max_delta = max(
        float(np.abs(
            probs[lf][s] - probs[lf][s-1]
        ).max())
        for s in range(t-window+2, t+1)
    )

    if max_delta > delta_limit:
        return False

    # 예측된 양성 증상의 확률 확인
    confidence = min(
        float(probs[lf][t][IDX[x]])
        for x in now
    )

    return confidence >= min_prob

experiments = {
    "guard2_d08_p65": (2, 0.08, 0.65),
    "guard3_d10_p60": (3, 0.10, 0.60),
}

for name, config in experiments.items():
    output = ROOT / f"{name}.csv"
    assert not output.exists(), f"기존 파일 보호: {output}"

    overrides = 0
    changed = 0

    with output.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label_file", "halt_step", "symptom"]
        )
        writer.writeheader()

        for lf in sorted(base):
            b = base[lf]
            l = learned[lf]

            bt = int(b["halt_step"])
            lt = int(l["halt_step"])

            # Learned가 먼저 중단 제안할 경우만 검사
            if lt < bt and safe_to_stop(lf, lt, *config):
                chosen = l
                overrides += 1
                if l["symptom"] != b["symptom"]:
                    changed += 1
            else:
                chosen = b

            writer.writerow(chosen)

    print("=" * 50)
    print("정책:", name)
    print("평가 통화:", len(base))
    print("조기 중단 허용:", overrides)
    print("증상 예측 변경:", changed)
    print("저장:", output)

print("\nGuard 실험 CSV 생성 완료!")
