import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from contextlib import ExitStack

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("Results/M42_RegDev")
ROOT.mkdir(parents=True, exist_ok=True)

PRED = Path("Results/V13A/early10_val.csv")
PROB = Path(
    "Results/M4_1/"
    "t5_t5_m41_v13a_targeted_3000_val_probs.jsonl"
)
BASE = Path("Results/M42_Pilot/base_3_d050.csv")
LABEL_DIR = Path("Work/OfficialScore/internal_val_labels")
DEV_LABEL_DIR = Path("Work/M42_RegDevLabels")

LABELS = [
    "고열", "구토", "두통", "복통", "어지러움",
    "열상", "오심", "전신쇠약", "호흡곤란"
]
IDX = {x: i for i, x in enumerate(LABELS)}

for path in [PRED, PROB, BASE]:
    assert path.is_file(), path
assert LABEL_DIR.is_dir()
DEV_LABEL_DIR.mkdir(parents=True, exist_ok=True)

def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def parse_symptoms(text):
    result = frozenset(
        s.strip() for s in text.split(";") if s.strip()
    )
    assert result.issubset(set(LABELS))
    return result

def sample_f1(pred, gold):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 1.0

# 1. 입력 로드
preds = defaultdict(dict)
calls = defaultdict(dict)

for r in load_csv(PRED):
    lf = r["label_file"]
    t = int(r["step"])
    assert t not in preds[lf]
    preds[lf][t] = (
        r["symptom"],
        parse_symptoms(r["symptom"])
    )

with PROB.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        lf = r["label_file"]
        t = int(r["step"])

        p = np.asarray(r["probs"], dtype=np.float32)
        y = np.asarray(r["target_vector"], dtype=np.int8)

        assert p.shape == y.shape == (9,)
        assert np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()
        assert t not in calls[lf]

        calls[lf][t] = (p, y)

base_rows = {
    r["label_file"]: r for r in load_csv(BASE)
}

assert len(preds) == len(calls) == len(base_rows) == 5840
assert preds.keys() == calls.keys() == base_rows.keys()

for lf, rows in calls.items():
    n = len(rows)
    expected = set(range(1, n + 1))
    assert set(rows) == expected
    assert set(preds[lf]) == expected
    assert 1 <= int(base_rows[lf]["halt_step"]) <= n

# 2. 동일한 평가 분할 유지
def hash_mod(text, mod):
    value = hashlib.sha256(text.encode()).hexdigest()[:8]
    return int(value, 16) % mod

test_calls = sorted(
    lf for lf in calls if hash_mod(lf, 4) == 0
)

dev_calls = sorted(
    lf for lf in calls
    if lf not in test_calls
    and hash_mod("reg-dev:" + lf, 5) == 0
)

train_calls = sorted(
    set(calls) - set(test_calls) - set(dev_calls)
)

assert len(test_calls) == 1431
assert len(train_calls) > 3000
assert len(dev_calls) > 600

print("학습 통화:", len(train_calls))
print("개발 통화:", len(dev_calls))
print("기존 평가 통화(미사용):", len(test_calls))

# 3. 기존 39개 Feature와 동일한 방식
def features(lf, t):
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

    result = np.concatenate([p0, d1, d2, extra])
    assert result.shape == (39,)
    return result

def candidates(lf):
    bt = int(base_rows[lf]["halt_step"])

    # 기존 Baseline보다 앞선 시점 중
    # 최근 두 예측이 동일한 경우만 후보
    return [
        t for t in range(6, bt)
        if preds[lf][t][1]
        and preds[lf][t][1] == preds[lf][t-1][1]
    ]

# 4. 회귀 학습 데이터
X, Y, W = [], [], []

for lf in train_calls:
    bt = int(base_rows[lf]["halt_step"])
    gold_vec = calls[lf][bt][1]

    gold = frozenset(
        LABELS[i] for i, v in enumerate(gold_vec) if v
    )

    baseline_f1 = sample_f1(
        parse_symptoms(base_rows[lf]["symptom"]), gold
    )

    ts = candidates(lf)
    if not ts:
        continue

    for t in ts:
        current_f1 = sample_f1(preds[lf][t][1], gold)

        X.append(features(lf, t))
        Y.append(current_f1 - baseline_f1)
        W.append(1.0 / len(ts))

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)
W = np.asarray(W, dtype=np.float32)

assert X.ndim == 2 and X.shape[1] == 39

print("\n===== 학습 데이터 검증 =====", flush=True)
print(f"학습 후보 수: {len(X):,}", flush=True)
print(f"Feature 수: {X.shape[1]}", flush=True)
print(f"목표값 평균: {Y.mean():.4f}", flush=True)
print(f"목표값 표준편차: {Y.std():.4f}", flush=True)
print(f"F1 개선 후보: {(Y > 0).sum():,}", flush=True)
print(f"F1 악화 후보: {(Y < 0).sum():,}", flush=True)
print(f"F1 동일 후보: {(Y == 0).sum():,}", flush=True)

assert len(X) >= 500, "학습 후보가 너무 적습니다."
assert Y.std() > 1e-6, "목표값 변동이 없습니다."

assert np.isfinite(Y).all()

W *= len(W) / W.sum()

print("학습 후보 시점:", len(X))
print("평균 실제 F1 변화:", round(float(Y.mean()), 5))

# 5. 회귀 모델 학습
model = HistGradientBoostingRegressor(
    max_iter=180,
    max_leaf_nodes=15,
    min_samples_leaf=100,
    learning_rate=0.05,
    l2_regularization=5.0,
    early_stopping=False,
    random_state=42
)

model.fit(X, Y, sample_weight=W)

model_path = ROOT / "delta_f1_regressor.pkl"
assert not model_path.exists(), "기존 모델 파일 보호"

joblib.dump({
    "model": model,
    "labels": LABELS,
    "target": "sample_f1_delta_vs_baseline"
}, model_path)

print("모델 저장:", model_path)

# 6. Dev 정답 파일 링크
for lf in dev_calls:
    assert Path(lf).name == lf

    src = (LABEL_DIR / lf).resolve(strict=True)
    dst = DEV_LABEL_DIR / lf

    if dst.is_symlink():
        assert dst.resolve() == src
    elif dst.exists():
        raise RuntimeError(f"파일 충돌: {dst}")
    else:
        dst.symlink_to(src)

# 7. 예측된 F1 변화량 기준 정책
thresholds = {
    "reg_m10": -0.10,
    "reg_m05": -0.05,
    "reg_p00": 0.00,
    "reg_p05": 0.05,
    "reg_p10": 0.10
}

names = ["baseline"] + list(thresholds)

for name in names:
    output = ROOT / f"{name}.csv"
    assert not output.exists(), f"기존 결과 보호: {output}"

stats = Counter()

with ExitStack() as stack:
    writers = {}

    for name in names:
        f = stack.enter_context(
            (ROOT / f"{name}.csv").open(
                "w", encoding="utf-8", newline=""
            )
        )
        w = csv.DictWriter(
            f,
            fieldnames=["label_file", "halt_step", "symptom"]
        )
        w.writeheader()
        writers[name] = w

    for lf in dev_calls:
        baseline = base_rows[lf]
        writers["baseline"].writerow(baseline)

        ts = candidates(lf)

        if ts:
            feats = np.stack([features(lf, t) for t in ts])
            predicted_delta = model.predict(feats)
        else:
            predicted_delta = np.empty(0)

        for name, threshold in thresholds.items():
            chosen = baseline

            # 시간 순서대로 최초 조건 만족 시 중단
            for t, value in zip(ts, predicted_delta):
                if value >= threshold:
                    chosen = {
                        "label_file": lf,
                        "halt_step": t,
                        "symptom": preds[lf][t][0]
                    }
                    stats[name] += 1
                    break

            writers[name].writerow(chosen)

print("\n===== Dev 중단 결과 =====")
for name in thresholds:
    print(name, "Baseline보다 빠른 중단:", stats[name])

print("\nDev CSV 생성 완료!")
print("결과 폴더:", ROOT)
print("정답 폴더:", DEV_LABEL_DIR)
