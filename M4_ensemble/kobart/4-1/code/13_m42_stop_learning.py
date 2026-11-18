import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np

# ==========================================
# 1. 경로 설정
# ==========================================

PRED = Path("Results/V13A/early10_val.csv")

PROB = Path(
    "Results/M4_1/"
    "t5_t5_m41_v13a_targeted_3000_val_probs.jsonl"
)

BASE = Path("Results/M42_Pilot/base_3_d050.csv")
LABEL_DIR = Path("Work/OfficialScore/internal_val_labels")

OUT = Path("Results/M42_Learned")
HOLDOUT = Path("Work/M42_LearnedHoldoutLabels")

OUT.mkdir(parents=True, exist_ok=True)
HOLDOUT.mkdir(parents=True, exist_ok=True)

LABELS = [
    "고열", "구토", "두통",
    "복통", "어지러움", "열상",
    "오심", "전신쇠약", "호흡곤란"
]
IDX = {s: i for i, s in enumerate(LABELS)}

for path in [PRED, PROB, BASE]:
    assert path.is_file(), path

assert LABEL_DIR.is_dir()

# ==========================================
# 2. 기존 증상 예측 읽기
# ==========================================

preds = defaultdict(dict)

with PRED.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for r in reader:
        lf = r["label_file"]
        t = int(r["step"])

        names = frozenset(
            x.strip()
            for x in r["symptom"].split(";")
            if x.strip()
        )

        assert names.issubset(set(LABELS))
        assert t not in preds[lf]

        preds[lf][t] = (r["symptom"], names)

# ==========================================
# 3. T5 확률 및 정답 읽기
# ==========================================

calls = defaultdict(dict)

with PROB.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)

        lf = r["label_file"]
        t = int(r["step"])

        p = np.asarray(r["probs"], dtype=np.float32)
        gold = np.asarray(
            r["target_vector"], dtype=np.int8
        )

        assert p.shape == gold.shape == (9,)
        assert np.all(np.isfinite(p))
        assert np.all((p >= 0) & (p <= 1))
        assert np.all((gold == 0) | (gold == 1))

        calls[lf][t] = (p, gold)

assert len(calls) == len(preds) == 5840
assert calls.keys() == preds.keys()

for lf, rows in calls.items():
    expected = set(range(1, len(rows) + 1))
    assert set(rows) == set(preds[lf]) == expected

# ==========================================
# 4. Feature 생성
# 현재 및 과거 발화만 사용
# ==========================================

def make_features(lf, t):
    rows = calls[lf]

    p0 = rows[t][0]
    p1 = rows[t-1][0] if t > 1 else p0
    p2 = rows[t-2][0] if t > 2 else p1

    d1 = np.abs(p0 - p1)
    d2 = np.abs(p1 - p2)

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
        float(np.min(np.abs(p0 - 0.5))),
        float(np.mean(np.abs(p0 - 0.5))),
        float(abs(p0[IDX["구토"]] - p0[IDX["오심"]])),
        float(p0[IDX["구토"]]),
        float(p0[IDX["오심"]])
    ], dtype=np.float32)

    return np.concatenate([
        p0, d1, d2, extra
    ]).astype(np.float32)


def base_stop(lf, t):
    if t < 6:
        return False

    a = preds[lf][t][1]
    b = preds[lf][t-1][1]
    c = preds[lf][t-2][1]

    if not a or not (a == b == c):
        return False

    p0 = calls[lf][t][0]
    p1 = calls[lf][t-1][0]
    p2 = calls[lf][t-2][0]

    delta = max(
        np.abs(p0-p1).max(),
        np.abs(p1-p2).max()
    )

    return delta <= 0.05

# ==========================================
# 5. 통화 단위 학습 / 평가 분리
# ==========================================

all_calls = sorted(calls)

test_calls = {
    lf for lf in all_calls
    if int(
        hashlib.sha256(lf.encode()).hexdigest()[:8], 16
    ) % 4 == 0
}

train_calls = [
    lf for lf in all_calls
    if lf not in test_calls
]

assert len(train_calls) > 4000
assert len(test_calls) > 1000

print("학습 통화:", len(train_calls), flush=True)
print("평가 통화:", len(test_calls), flush=True)

# ==========================================
# 6. STOP 신뢰도 학습 데이터
# ==========================================

X, y, weights = [], [], []

for lf in train_calls:
    rows = calls[lf]

    # 통화별 가중치 균형
    weight = 1.0 / max(1, len(rows)-5)

    for t in range(6, len(rows)+1):
        feat = make_features(lf, t)

        gold = frozenset(
            LABELS[i]
            for i, v in enumerate(rows[t][1])
            if v == 1
        )

        predicted = preds[lf][t][1]

        X.append(feat)
        y.append(int(predicted == gold))
        weights.append(weight)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int8)
weights = np.asarray(weights, dtype=np.float32)

assert len(X) > 10000
assert len(np.unique(y)) == 2

weights *= len(weights) / weights.sum()

print("학습 시점:", len(X), flush=True)
print("정확한 증상 집합 비율:", round(y.mean(), 4), flush=True)

# ==========================================
# 7. LightGBM 학습
# 미설치 시 sklearn 대체
# ==========================================

try:
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=120,
        reg_lambda=5,
        n_jobs=4,
        random_state=42,
        verbosity=-1
    )
    model_name = "LightGBM"

except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=110,
        learning_rate=0.07,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=5,
        random_state=42
    )
    model_name = "HistGradientBoosting"

print("학습 모델:", model_name, flush=True)

model.fit(X, y, sample_weight=weights)

import joblib

joblib.dump(
    {"model": model, "labels": LABELS},
    OUT / "stop_model.pkl"
)

print("모델 학습 및 저장 완료!", flush=True)

# ==========================================
# 8. 기존 Baseline 로드
# ==========================================

baseline = {}

with BASE.open(
    encoding="utf-8-sig", newline=""
) as f:
    for r in csv.DictReader(f):
        baseline[r["label_file"]] = r

assert len(baseline) == 5840

# ==========================================
# 9. 평가용 정답 폴더 생성
# 학습에 사용하지 않은 통화만 링크
# ==========================================

for lf in sorted(test_calls):
    assert Path(lf).name == lf

    src = (LABEL_DIR / lf).resolve(strict=True)
    dst = HOLDOUT / lf

    if dst.is_symlink():
        assert dst.resolve() == src
    elif dst.exists():
        raise RuntimeError(f"파일 충돌: {dst}")
    else:
        dst.symlink_to(src)

# ==========================================
# 10. 중단 정책 비교
# ==========================================

names = [
    "baseline",
    "learned075",
    "learned085",
    "learned090",
    "hybrid070"
]

files = {
    name: (OUT / f"{name}.csv").open(
        "w", encoding="utf-8", newline=""
    )
    for name in names
}

writers = {
    name: csv.writer(f)
    for name, f in files.items()
}

for writer in writers.values():
    writer.writerow([
        "label_file", "halt_step", "symptom"
    ])

stop_stats = defaultdict(list)

try:
    for lf in sorted(test_calls):
        last = len(calls[lf])

        # 기존 정책 결과
        b = baseline[lf]

        writers["baseline"].writerow([
            lf, b["halt_step"], b["symptom"]
        ])
        stop_stats["baseline"].append(
            int(b["halt_step"])
        )

        # 현재 시점별 Stop Probability 예측
        times = list(range(6, last+1))

        if times:
            features = np.stack([
                make_features(lf, t) for t in times
            ])

            scores = model.predict_proba(features)[:, 1]
        else:
            scores = []

        decisions = {
            name: last
            for name in names
            if name != "baseline"
        }

        for t, score in zip(times, scores):
            if not preds[lf][t][1]:
                continue

            if decisions["learned075"] == last and score >= 0.75:
                decisions["learned075"] = t

            if decisions["learned085"] == last and score >= 0.85:
                decisions["learned085"] = t

            if decisions["learned090"] == last and score >= 0.90:
                decisions["learned090"] = t

            if (
                decisions["hybrid070"] == last
                and score >= 0.70
                and base_stop(lf, t)
            ):
                decisions["hybrid070"] = t

        for name, t in decisions.items():
            writers[name].writerow([
                lf, t, preds[lf][t][0]
            ])
            stop_stats[name].append(t)

finally:
    for f in files.values():
        f.close()

# ==========================================
# 11. 결과
# ==========================================

print("\n===== 실험 CSV 생성 결과 =====")

for name in names:
    steps = stop_stats[name]
    print(
        name,
        "통화:", len(steps),
        "평균 중단:", round(float(np.mean(steps)), 2)
    )

print("\n평가 정답 폴더:", HOLDOUT)
print("결과 폴더:", OUT)
print("전체 작업 완료!")
