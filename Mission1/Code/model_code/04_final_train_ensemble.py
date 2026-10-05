"""
04_final_train_ensemble.py

역할: baseline(raw+LogReg) -> L2정규화+MLP 교체 -> duration 가중평균 집계
      -> 10-seed 앙상블 -> 통계검증(McNemar)으로 이어지는 실험 과정에서,
      최종 채택된 구성만 골라 하나의 스크립트로 재현.

구성: L2 정규화 + StandardScaler + MLP(hidden=128) x 10 seed 앙상블
      + duration 가중평균 집계

Decision Threshold: 0.5로 고정. 

Probability Calibration: 규정 Q4에 따라 확률값 자체를 보정하는 것은 허용되므로,
      val_calls(학습에 전혀 쓰이지 않은 held-out)로 Platt scaling을 학습해서
      Data/Validation에서 실제로 개선되는지 확인. 이 보정 이후에도 Decision
      Threshold는 여전히 0.5로 고정 (보정은 확률값을 바꾸는 것일 뿐 cutoff 이동이 아님).

실행하면:
  1. 이미 추출된 임베딩(shard)을 로드 (재추출 없음)
  2. baseline(raw+LogReg) 학습 - 비교 기준용
  3. MLP를 seed 10개로 학습해 앙상블 구성
  4. threshold 0.5 고정으로 val_calls / Data/Validation 정확도 평가
  5. McNemar 검정으로 baseline 대비 개선의 통계적 유의성 확인
  6. val_calls로 Platt scaling 학습 후 calibration 전/후 정확도 비교
  7. 최종 모델을 .ckpt로 저장

실험 과정(왜 이 구성을 택했는지의 근거: seed 안정성, 단일모델과의 비교 등)은
Mission1_학습노트북.ipynb에 별도로 정리되어 있음.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score
from scipy.stats import binomtest


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_SHARD_DIR = BASE_DIR / "Work" / "Common" / "embeddings" / "shards"
PV_SHARD_DIR = BASE_DIR / "Work" / "ProvidedValidation" / "Common" / "embeddings" / "shards"
MODEL_OUT_DIR = BASE_DIR / "Work" / "Common" / "models"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 최종 채택된 구성 (근거는 Mission1_학습노트북.ipynb 참고) ----
DURATION_WEIGHT_CAP_MS = 3000
SEEDS = list(range(10))          # 10개 seed 평균 -> 단일 seed 선택 시의 편향 회피
MLP_HIDDEN = (128,)               # LogReg 대비 가장 크게 기여한 변경
PREPROC_KIND = "l2"               # 임베딩 L2 정규화

# Decision Threshold는 0.5로 고정 
DECISION_THRESHOLD = 0.5

# 확률 calibration 사용 여부: val_calls로 Platt scaling을 학습해서
# Data/Validation에서도 실제로 개선되는지 확인한 뒤 True/False를 결정할 것 —
# 아래 실행 로그의 "calibration 후" Data/Validation 정확도를 보고 판단.
USE_PROBABILITY_CALIBRATION = True


def load_shards(shard_dir):
    npy_files = sorted(shard_dir.glob("shard_*.npy"))
    csv_files = sorted(shard_dir.glob("shard_*.csv"))
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
    return X, meta


def aggregate_duration_weighted(meta_df, proba, threshold):
    df = meta_df.copy()
    df["proba"] = proba
    w = df["duration_ms"].clip(upper=DURATION_WEIGHT_CAP_MS)
    df["w_proba"] = df["proba"] * w
    grouped = df.groupby("call_id").agg(
        w_sum=("w_proba", "sum"),
        w_total=("duration_ms", lambda s: s.clip(upper=DURATION_WEIGHT_CAP_MS).sum()),
    )
    call_proba = grouped["w_sum"] / grouped["w_total"]
    call_true = df.groupby("call_id")["gender"].first().map({"M": 0, "F": 1})
    call_proba = call_proba.reindex(call_true.index)
    call_pred = (call_proba >= threshold).astype(int)
    return call_true, call_pred, call_proba


def aggregate_mean(meta_df, proba, threshold):
    df = meta_df.copy()
    df["proba"] = proba
    call_proba = df.groupby("call_id")["proba"].mean()
    call_true = df.groupby("call_id")["gender"].first().map({"M": 0, "F": 1})
    call_proba = call_proba.reindex(call_true.index)
    call_pred = (call_proba >= threshold).astype(int)
    return call_true, call_pred


def mcnemar_exact(pred_a, pred_b, y_true):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(((~correct_a) & correct_b).sum())
    c = int((correct_a & (~correct_b)).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    result = binomtest(min(b, c), n, 0.5, alternative="two-sided")
    return b, c, result.pvalue


def to_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt_calibrator(call_proba, call_true):
    """val_calls(학습에 쓰이지 않은 held-out)의 통화 단위 확률/정답으로 Platt
    scaling(1-feature 로지스틱 회귀)을 학습. 규정 Q4: 확률 자체를 보정하는 것은
    허용되며, 이 보정은 Decision Threshold를 옮기는 것이 아니라 확률값을 바꾸는 것."""
    clf = LogisticRegression(C=1e6, max_iter=1000)
    clf.fit(to_logit(call_proba).reshape(-1, 1), call_true)
    return clf


def apply_calibration(calibrator, call_proba):
    return calibrator.predict_proba(to_logit(call_proba).reshape(-1, 1))[:, 1]


def main():
    # ---- 데이터 로드 (이미 추출된 임베딩, 재추출 없음) ----
    print("shard 로드 중...")
    X_all, meta_all = load_shards(TRAIN_SHARD_DIR)
    train_mask = (meta_all["split"] == "train").to_numpy()
    val_mask = (meta_all["split"] == "val").to_numpy()
    X_train, meta_train = X_all[train_mask], meta_all[train_mask].reset_index(drop=True)
    X_val, meta_val = X_all[val_mask], meta_all[val_mask].reset_index(drop=True)
    y_train = (meta_train["gender"] == "F").astype(int).to_numpy()

    pv_available = PV_SHARD_DIR.exists() and any(PV_SHARD_DIR.glob("shard_*.npy"))
    if pv_available:
        X_pv, meta_pv = load_shards(PV_SHARD_DIR)
    print(f"Train {len(y_train)} / Val {len(meta_val)}"
          + (f" / Data-Validation {len(meta_pv)}" if pv_available else ""))

    # ---- Baseline (비교 기준, threshold 0.5) ----
    print("\nBaseline (raw + LogisticRegression) 학습 중...")
    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(X_train)
    X_val_base = scaler_base.transform(X_val)
    clf_base = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf_base.fit(X_train_base, y_train)
    val_proba_base = clf_base.predict_proba(X_val_base)[:, 1]
    val_true, val_pred_base = aggregate_mean(meta_val, val_proba_base, threshold=DECISION_THRESHOLD)
    print(f"  Baseline val_calls Accuracy: {accuracy_score(val_true, val_pred_base)*100:.2f}%")

    if pv_available:
        X_pv_base = scaler_base.transform(X_pv)
        pv_proba_base = clf_base.predict_proba(X_pv_base)[:, 1]
        pv_true, pv_pred_base = aggregate_mean(meta_pv, pv_proba_base, threshold=DECISION_THRESHOLD)
        print(f"  Baseline Data/Validation Accuracy: {accuracy_score(pv_true, pv_pred_base)*100:.2f}%")

    # ---- 최종 구성: L2정규화 + MLP 10-seed 앙상블 ----
    print(f"\n최종 구성(L2 + MLP{MLP_HIDDEN}) seed {SEEDS}로 학습 중...")

    X_train_l2 = normalize(X_train, norm="l2")
    X_val_l2 = normalize(X_val, norm="l2")
    if pv_available:
        X_pv_l2 = normalize(X_pv, norm="l2")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_l2)
    X_val_s = scaler.transform(X_val_l2)
    if pv_available:
        X_pv_s = scaler.transform(X_pv_l2)

    val_proba_list, pv_proba_list, fitted_clfs = [], [], []

    for seed in SEEDS:
        clf = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN, activation="relu", solver="adam",
            alpha=1e-4, max_iter=150, early_stopping=True, n_iter_no_change=10,
            random_state=seed,
        )
        clf.fit(X_train_s, y_train)
        val_proba_list.append(clf.predict_proba(X_val_s)[:, 1])
        if pv_available:
            pv_proba_list.append(clf.predict_proba(X_pv_s)[:, 1])
        fitted_clfs.append(clf)
        print(f"  seed={seed} 완료")

    val_proba_ens = np.mean(val_proba_list, axis=0)

    # ---- Decision Threshold 0.5 고정으로 평가 ----
    val_true_full, val_pred_ens, val_call_proba = aggregate_duration_weighted(
        meta_val, val_proba_ens, DECISION_THRESHOLD
    )
    print(f"\nDecision Threshold: {DECISION_THRESHOLD} (대회 규정상 고정값, 탐색하지 않음)")
    print(f"val_calls Accuracy: {accuracy_score(val_true_full, val_pred_ens)*100:.2f}%")

    if pv_available:
        pv_proba_ens = np.mean(pv_proba_list, axis=0)
        pv_true_full, pv_pred_ens, pv_call_proba = aggregate_duration_weighted(
            meta_pv, pv_proba_ens, DECISION_THRESHOLD
        )
        print(f"Data/Validation Accuracy: {accuracy_score(pv_true_full, pv_pred_ens)*100:.2f}%")

    # ---- McNemar 검정 (baseline 대비 유의성) ----
    print("\nMcNemar 검정 (baseline vs 최종 앙상블)")
    b, c, p = mcnemar_exact(val_pred_base.to_numpy(), val_pred_ens.to_numpy(), val_true_full.to_numpy())
    print(f"  val_calls: 고침 {b} / 새로틀림 {c}  (p={p:.6f}) ->", "유의미" if p < 0.05 else "유의성 부족")
    if pv_available:
        b, c, p = mcnemar_exact(pv_pred_base.to_numpy(), pv_pred_ens.to_numpy(), pv_true_full.to_numpy())
        print(f"  Data/Validation: 고침 {b} / 새로틀림 {c}  (p={p:.6f}) ->", "유의미" if p < 0.05 else "유의성 부족")

    # ---- Probability Calibration (Platt scaling) ----
    print("\nProbability Calibration (Platt scaling, val_calls로 학습)")
    calibrator = fit_platt_calibrator(val_call_proba.to_numpy(), val_true_full.to_numpy())

    val_proba_cal = apply_calibration(calibrator, val_call_proba.to_numpy())
    val_pred_cal = (val_proba_cal >= DECISION_THRESHOLD).astype(int)
    print(f"  val_calls Accuracy       calibration 전: {accuracy_score(val_true_full, val_pred_ens)*100:.2f}%"
          f"  /  calibration 후: {accuracy_score(val_true_full, val_pred_cal)*100:.2f}% (학습에 쓰인 데이터라 참고용)")

    if pv_available:
        pv_proba_cal = apply_calibration(calibrator, pv_call_proba.to_numpy())
        pv_pred_cal = (pv_proba_cal >= DECISION_THRESHOLD).astype(int)
        print(f"  Data/Validation Accuracy calibration 전: {accuracy_score(pv_true_full, pv_pred_ens)*100:.2f}%"
              f"  /  calibration 후: {accuracy_score(pv_true_full, pv_pred_cal)*100:.2f}%  <- 실제 효과 확인 지점")

    print(f"\nUSE_PROBABILITY_CALIBRATION = {USE_PROBABILITY_CALIBRATION} 로 최종 저장"
          " (위 Data/Validation calibration 전/후 수치를 보고 이 값을 True/False로 정할 것)")

    # ---- 최종 모델을 .ckpt로 저장 ----
    out_path = MODEL_OUT_DIR / "ecapa_ensemble_v3.ckpt"
    torch.save(
        {
            "scaler": scaler,
            "clfs": fitted_clfs,
            "preproc_kind": PREPROC_KIND,
            "weight_mode": "duration",
            "duration_weight_cap_ms": DURATION_WEIGHT_CAP_MS,
            "threshold": float(DECISION_THRESHOLD),
            "seeds": SEEDS,
            "calibrator": calibrator if USE_PROBABILITY_CALIBRATION else None,
            "use_calibration": bool(USE_PROBABILITY_CALIBRATION),
        },
        out_path,
    )

    print("\n최종 모델 저장 위치:", out_path)
    print("저장된 설정: threshold =", DECISION_THRESHOLD, " / calibration =", USE_PROBABILITY_CALIBRATION)


if __name__ == "__main__":
    main()
