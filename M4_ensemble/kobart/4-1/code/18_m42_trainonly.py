import json
import hashlib
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# ==========================================
# Mission 4-2: Train-only Stop Regression
# Validation 정답은 절대 사용하지 않음
# ==========================================

SOURCE = Path(
    "Results/M4_1/"
    "t5_t5_m41_v13a_targeted_3000_train_probs.jsonl"
)

OUT = Path("Results/M42_TrainOnly")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT / "delta_f1_proxy10_v1.pkl"
REPORT_PATH = OUT / "train_dev_report_v1.json"

assert SOURCE.is_file(), f"파일 없음: {SOURCE}"
assert not MODEL_PATH.exists(), "기존 모델 보호"
assert not REPORT_PATH.exists(), "기존 리포트 보호"

LABELS = [
    "고열", "구토", "두통", "복통",
    "어지러움", "열상", "오심",
    "전신쇠약", "호흡곤란"
]

# ==========================================
# 1. Train 데이터만 불러오기
# ==========================================

calls = defaultdict(dict)
targets = {}

with SOURCE.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)

        lf = r["label_file"]
        t = int(r["step"])

        p = np.asarray(r["probs"], dtype=np.float32)
        y = np.asarray(
            r["target_vector"], dtype=np.int8
        )

        assert p.shape == y.shape == (9,)
        assert np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()
        assert ((y == 0) | (y == 1)).all()
        assert t not in calls[lf]

        if lf in targets:
            assert np.array_equal(targets[lf], y)
        else:
            targets[lf] = y

        calls[lf][t] = p

print("Train 전체 통화:", len(calls))

# ==========================================
# 2. F1 계산
# ==========================================

def indices(p):
    return frozenset(
        np.flatnonzero(p >= 0.5).tolist()
    )

def f1(pred, gold):
    if not pred and not gold:
        return 1.0

    tp = len(pred & gold)
    d = len(pred) + len(gold)

    return 2 * tp / d if d else 0.0

# ==========================================
# 3. 기존 39개 Feature 구성
# ==========================================

def features(lf, t):
    rows = calls[lf]

    p0 = rows[t]
    p1 = rows[t-1]
    p2 = rows[t-2]

    d1 = np.abs(p0-p1)
    d2 = np.abs(p1-p2)

    now = indices(p0)
    prev = indices(p1)
    prev2 = indices(p2)

    active = list(now)

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
        float(abs(p0[1]-p0[6])),
        float(p0[1]),
        float(p0[6])
    ], dtype=np.float32)

    x = np.concatenate([p0, d1, d2, extra])
    assert x.shape == (39,)

    return x

# ==========================================
# 4. Train 조기 중단 후보 생성
# ==========================================

def get_examples(lf):
    rows = calls[lf]

    # 10번째 예측을 비교 기준으로 사용
    if 10 not in rows:
        return []

    gold = frozenset(
        np.flatnonzero(targets[lf]).tolist()
    )

    # 정답 증상 목록이 빈 샘플은 제외
    if not gold:
        return []

    reference_f1 = f1(indices(rows[10]), gold)

    result = []

    # Train은 중간 Step 누락이 있어
    # 연속 시점 6~8만 사용
    for t in (6, 7, 8):
        if not all(
            s in rows for s in (t, t-1, t-2)
        ):
            continue

        now = indices(rows[t])
        prev = indices(rows[t-1])

        if not now or now != prev:
            continue

        current_f1 = f1(now, gold)
        delta = current_f1 - reference_f1

        result.append((
            t,
            features(lf, t),
            delta
        ))

    return result

# ==========================================
# 5. Train 내부 분할 (통화 단위)
# ==========================================

def is_dev(lf):
    key = "m42-trainonly-v1:" + lf
    digest = hashlib.sha256(
        key.encode()
    ).digest()

    return int.from_bytes(
        digest[:8], "big"
    ) % 5 == 0

X, Y, W = [], [], []
dev_data = {}

train_count = 0
dev_count = 0

for lf in sorted(calls):
    examples = get_examples(lf)

    if not examples:
        continue

    if is_dev(lf):
        dev_data[lf] = examples
        dev_count += 1
    else:
        train_count += 1

        for t, x, delta in examples:
            X.append(x)
            Y.append(delta)
            W.append(1.0 / len(examples))

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)
W = np.asarray(W, dtype=np.float32)

assert X.ndim == 2 and X.shape[1] == 39
assert len(X) >= 500
assert len(dev_data) >= 100
assert Y.std() > 1e-6

W *= len(W) / W.sum()

print("\n===== Train 데이터 검증 =====")
print("학습 통화:", train_count)
print("개발 통화:", dev_count)
print("학습 후보:", len(X))
print("F1 개선 후보:", int((Y > 0).sum()))
print("F1 악화 후보:", int((Y < 0).sum()))
print("F1 동일 후보:", int((Y == 0).sum()))

# ==========================================
# 6. 회귀 모델 학습
# ==========================================

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

# ==========================================
# 7. Train 내부 Dev 평가
# ==========================================

threshold = 0.0

selected = 0
deltas = []

for lf, examples in dev_data.items():
    x = np.stack([
        item[1] for item in examples
    ])

    predictions = model.predict(x)

    chosen_delta = 0.0

    # 시간순 최초 조건 만족 시 중단
    for item, prediction in zip(
        examples, predictions
    ):
        t, feature, actual_delta = item

        if prediction >= threshold:
            chosen_delta = float(actual_delta)
            selected += 1
            break

    deltas.append(chosen_delta)

mean_delta = float(np.mean(deltas))

print("\n===== Train 내부 Dev 결과 =====")
print("개발 통화:", len(dev_data))
print("조기 중단 제안:", selected)
print(
    "평균 Sample F1 변화:",
    round(mean_delta, 5)
)

# ==========================================
# 8. 모델 저장
# ==========================================

joblib.dump({
    "model": model,
    "labels": LABELS,
    "feature_size": 39,
    "train_only": True,
    "reference_step": 10
}, MODEL_PATH)

report = {
    "source": str(SOURCE),
    "validation_labels_used": False,
    "train_calls": train_count,
    "dev_calls": dev_count,
    "train_examples": len(X),
    "threshold": threshold,
    "dev_early_selected": selected,
    "dev_mean_delta_vs_step10": mean_delta,
    "target": "sample_f1_delta_vs_step10"
}

REPORT_PATH.write_text(
    json.dumps(
        report, ensure_ascii=False, indent=2
    ),
    encoding="utf-8"
)

print("\n===== 학습 완료 =====")
print("모델:", MODEL_PATH)
print("결과:", REPORT_PATH)
print("Validation 학습 사용: 없음")
