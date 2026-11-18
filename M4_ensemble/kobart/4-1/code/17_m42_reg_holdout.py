import csv
import json
import hashlib
import joblib
import numpy as np

from pathlib import Path
from collections import defaultdict

ROOT = Path("Results/M42_Learned")

MODEL = Path(
    "Results/M42_RegDev/delta_f1_regressor.pkl"
)

PRED = Path("Results/V13A/early10_val.csv")

PROB = Path(
    "Results/M4_1/"
    "t5_t5_m41_v13a_targeted_3000_val_probs.jsonl"
)

BASE = ROOT / "baseline.csv"
OUT = ROOT / "reg_p00_holdout_v1.csv"

LABELS = [
    "고열", "구토", "두통", "복통",
    "어지러움", "열상", "오심",
    "전신쇠약", "호흡곤란"
]

IDX = {x: i for i, x in enumerate(LABELS)}

for p in [MODEL, PRED, PROB, BASE]:
    assert p.is_file(), p

assert not OUT.exists(), f"기존 파일 보호: {OUT}"

saved = joblib.load(MODEL)
model = saved["model"]

assert saved["labels"] == LABELS
assert model.n_features_in_ == 39

# Baseline 평가 통화
with BASE.open(encoding="utf-8-sig", newline="") as f:
    base = {
        r["label_file"]: r
        for r in csv.DictReader(f)
    }

assert len(base) == 1431

# 이전에 고정한 평가 분할인지 확인
for lf in base:
    h = hashlib.sha256(lf.encode()).hexdigest()[:8]
    assert int(h, 16) % 4 == 0

preds = defaultdict(dict)
probs = defaultdict(dict)

with PRED.open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        lf = r["label_file"]
        if lf not in base:
            continue

        t = int(r["step"])
        assert t not in preds[lf]

        names = frozenset(
            x.strip()
            for x in r["symptom"].split(";")
            if x.strip()
        )

        assert names.issubset(set(LABELS))
        preds[lf][t] = (r["symptom"], names)

with PROB.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        lf = r["label_file"]

        if lf not in base:
            continue

        t = int(r["step"])
        assert t not in probs[lf]

        p = np.asarray(r["probs"], dtype=np.float32)
        assert p.shape == (9,)
        assert np.isfinite(p).all()

        probs[lf][t] = p

assert set(preds) == set(probs) == set(base)

def features(lf, t):
    rows = probs[lf]

    p0 = rows[t]
    p1 = rows[t-1] if t > 1 else p0
    p2 = rows[t-2] if t > 2 else p1

    d1 = np.abs(p0-p1)
    d2 = np.abs(p1-p2)

    now = preds[lf][t][1]
    prev = preds[lf][t-1][1] if t > 1 else frozenset()
    prev2 = preds[lf][t-2][1] if t > 2 else frozenset()

    active = [IDX[x] for x in now]

    extra = np.array([
        min(t, 100),
        len(now),
        int(now == prev),
        int(now == prev == prev2),
        float(d1.max()),
        float(max(d1.max(), d2.max())),
        float(np.min(p0[active])) if active else 0,
        float(np.min(np.abs(p0-0.5))),
        float(np.mean(np.abs(p0-0.5))),
        float(abs(p0[IDX["구토"]]-p0[IDX["오심"]])),
        float(p0[IDX["구토"]]),
        float(p0[IDX["오심"]])
    ], dtype=np.float32)

    x = np.concatenate([p0, d1, d2, extra])
    assert x.shape == (39,)
    return x

# 중단 정책
changed = 0
halt_steps = []

with OUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["label_file", "halt_step", "symptom"]
    )
    writer.writeheader()

    for lf in sorted(base):
        baseline = base[lf]
        bt = int(baseline["halt_step"])

        n = len(probs[lf])
        expected = set(range(1, n+1))

        assert set(probs[lf]) == expected
        assert set(preds[lf]) == expected
        assert 1 <= bt <= n

        chosen = baseline

        # Baseline 중단 이전 시점만 검사
        for t in range(6, bt):
            now = preds[lf][t][1]
            prev = preds[lf][t-1][1]

            if not now or now != prev:
                continue

            x = features(lf, t).reshape(1, -1)

            predicted_delta = float(model.predict(x)[0])

            # reg_p00: 예상 F1 변화량 >= 0
            if predicted_delta >= 0.0:
                chosen = {
                    "label_file": lf,
                    "halt_step": t,
                    "symptom": preds[lf][t][0]
                }
                changed += 1
                break

        writer.writerow(chosen)
        halt_steps.append(int(chosen["halt_step"]))

print("=" * 50)
print("reg_p00 Holdout 검증 CSV 생성 완료")
print("=" * 50)
print("평가 통화:", len(base))
print("기존보다 빠른 중단:", changed)
print("평균 중단 시점:", round(np.mean(halt_steps), 2))
print("저장:", OUT)
