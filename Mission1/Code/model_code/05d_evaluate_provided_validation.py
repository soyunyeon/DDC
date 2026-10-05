"""
05d_evaluate_provided_validation.py

04_final_train_ensemble.py에서 학습·저장해둔 최종 모델
(Work\\Common\\models\\ecapa_ensemble_v3.ckpt)을 그대로 불러와서,
Data\\Validation(대회 제공 검증셋)에 대해 "예측만" 수행하고 정확도를 확인.

-> 재학습 없음 (같은 frozen ECAPA 임베딩 + 이미 학습된 MLP 10-seed 앙상블).
-> segment 단위 / 통화(call) 단위(duration 가중평균 + threshold) 둘 다 출력해서
   기존 결과(val_calls.csv 기준 98.10%)와 바로 비교 가능.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Work" / "Common" / "models" / "ecapa_ensemble_v3.ckpt"
PV_SHARD_DIR = BASE_DIR / "Work" / "ProvidedValidation" / "Common" / "embeddings" / "shards"

DURATION_WEIGHT_CAP_MS = 3000


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"학습된 모델이 없습니다: {MODEL_PATH}\n"
        f"04_final_train_ensemble.py를 먼저 실행해서 모델을 저장하세요."
    )

print("모델 로드:", MODEL_PATH)
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
scaler = ckpt["scaler"]
clfs = ckpt["clfs"]
preproc_kind = ckpt.get("preproc_kind", "l2")
threshold = ckpt.get("threshold", 0.5)

print(f"분류기 {len(clfs)}개, preproc={preproc_kind}, threshold={threshold}")


npy_files = sorted(PV_SHARD_DIR.glob("shard_*.npy"))
csv_files = sorted(PV_SHARD_DIR.glob("shard_*.csv"))

if not npy_files:
    raise FileNotFoundError(
        f"shard 파일이 없습니다: {PV_SHARD_DIR}\n"
        f"05c_extract_embeddings_provided_validation.py를 먼저 실행하세요."
    )

print("shard 개수:", len(npy_files))

X_list, meta_list = [], []
for npy_path, csv_path in zip(npy_files, csv_files):
    x = np.load(npy_path)
    if x.shape[0] == 0:
        continue
    m = pd.read_csv(csv_path)
    X_list.append(x)
    meta_list.append(m)

X = np.concatenate(X_list, axis=0)
meta = pd.concat(meta_list, axis=0).reset_index(drop=True)

print("전체 임베딩 shape:", X.shape, " 메타데이터 행 수:", len(meta))
assert X.shape[0] == len(meta), "임베딩 개수와 메타데이터 행 수가 다릅니다."


# F=1, M=0
y_true = (meta["gender"] == "F").astype(int).to_numpy()

Xp = normalize(X, norm="l2") if preproc_kind == "l2" else X
Xp = scaler.transform(Xp)

# 10-seed 앙상블: 각 분류기의 확률을 평균
proba_stack = np.stack([clf.predict_proba(Xp)[:, 1] for clf in clfs], axis=0)
proba = proba_stack.mean(axis=0)
pred = (proba >= threshold).astype(int)

seg_acc = accuracy_score(y_true, pred)

print()
print("=" * 60)
print("Segment 단위 평가 (Data\\Validation, 대회 제공 검증셋)")
print("=" * 60)
print("Accuracy:", round(seg_acc, 4))
print(classification_report(y_true, pred, target_names=["M", "F"]))
print("Confusion Matrix (행=실제, 열=예측, 순서 M,F):\n", confusion_matrix(y_true, pred))


meta = meta.copy()
meta["proba"] = proba

# 통화 단위 집계: duration 가중평균 (최종 채택 방식)
w = meta["duration_ms"].clip(upper=DURATION_WEIGHT_CAP_MS)
meta["w_proba"] = meta["proba"] * w
grouped = meta.groupby("call_id").agg(
    w_sum=("w_proba", "sum"),
    w_total=("duration_ms", lambda s: s.clip(upper=DURATION_WEIGHT_CAP_MS).sum()),
)
call_proba = grouped["w_sum"] / grouped["w_total"]
call_true = meta.groupby("call_id")["gender"].first().map({"M": 0, "F": 1})
call_proba = call_proba.reindex(call_true.index)
call_pred = (call_proba >= threshold).astype(int)

acc_call = accuracy_score(call_true, call_pred)

print()
print("=" * 60)
print(f"통화(call) 단위 평가 (Data\\Validation, 통화 수: {len(call_true)})")
print("=" * 60)
print(f"duration 가중평균 + threshold={threshold} Accuracy:", round(acc_call, 4))

print()
print("=" * 60)
print("참고: val_calls.csv(Training 파티션 내 held-out) 결과(98.10%)와 비교해보세요.")
print("두 숫자가 비슷하면 일반화가 잘 되고 있다는 좋은 신호,")
print("Data\\Validation 쪽이 뚜렷이 낮다면 val_calls.csv 성능이 다소 낙관적이었을 가능성.")
print("=" * 60)
