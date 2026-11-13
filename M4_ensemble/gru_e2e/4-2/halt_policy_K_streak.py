# ============================================================
# 4-2 halt 정책 오프라인 시뮬레이션
#
# inference_gru.py --mission 4-1 실행 결과로 나오는 *_probs.csv
# (매 시점 9개 클래스 raw 확률)를 가지고, 모델을 다시 돌리지 않고
# "언제 멈출지"만 바꿔가며 mission4-2.csv 형식의 결과를 만든다.
#
# halt 정책: threshold는 주최측 공지대로 0.5 고정. "언제 멈출지"는
# 0.5로 판정한 증상 집합이 최근 K개 시점 동안 안 바뀌면 멈추는
# 안정성 규칙을 쓴다 (별도의 confidence threshold를 새로 만드는 게
# 아니라, 이미 고정된 0.5 판정 결과의 "안정성"만 보는 방식이라 공지
# 2번 규정과 충돌하지 않는다).
#
# causal 보장: 각 시점의 판단은 그 시점까지의 데이터만 본다 (앞으로
# 스캔하면서 지금까지의 연속 동일 판정 횟수만 세므로 미래를 보지 않음).
# 끝까지 안정되지 않으면 마지막 시점에서 멈춘다 (= 끝까지 듣는 것과 동일).
#
# 사용법
#   python 05_simulate_halt.py \
#       --probs ../outputs/mission4-1_valholdout_probs.csv \
#       --output ../outputs/mission4-2_sim_k3.csv --k 3
# ============================================================

import argparse
import csv
from collections import defaultdict

SYMPTOM_CLASSES = [
    "고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란",
]
THRESHOLD = 0.5  # 주최측 공지 고정값


def load_probs(path):
    """label_file -> [(step, {symptom: prob}), ...] (step 오름차순)"""
    calls = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            probs = {c: float(row[c]) for c in SYMPTOM_CLASSES}
            calls[row["label_file"]].append((step, probs))
    for lf in calls:
        calls[lf].sort(key=lambda x: x[0])
    return calls


def pred_set(probs):
    return frozenset(c for c in SYMPTOM_CLASSES if probs[c] > THRESHOLD)


def simulate_call(steps, k):
    """
    steps: [(step, probs), ...] 시간순.
    최근 k개 시점의 예측 집합이 전부 동일해지는 첫 시점에서 멈춘다.
    끝까지 안정 안 되면 마지막 시점에서 멈춘다 (listen-to-end fallback).
    """
    prev_set = None
    streak = 0
    for step, probs in steps:
        cur_set = pred_set(probs)
        if cur_set == prev_set:
            streak += 1
        else:
            streak = 1
        prev_set = cur_set
        if streak >= k:
            return step, cur_set
    # 안정 안 됨 -> 마지막 시점
    last_step, last_probs = steps[-1]
    return last_step, pred_set(last_probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="inference_gru.py가 만든 *_probs.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", type=int, default=3, help="연속 몇 시점 안정되면 멈출지")
    args = ap.parse_args()

    calls = load_probs(args.probs)
    print(f"[info] 통화 {len(calls)}건, k={args.k}")

    rows = []
    halt_steps = []
    for label_file, steps in calls.items():
        halt_step, final_set = simulate_call(steps, args.k)
        symptom_str = ";".join(c for c in SYMPTOM_CLASSES if c in final_set)
        rows.append((label_file, halt_step, symptom_str))
        halt_steps.append(halt_step)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label_file", "halt_step", "symptom"])
        w.writerows(rows)

    avg_halt = sum(halt_steps) / len(halt_steps) if halt_steps else 0
    print(f"[done] {args.output} ({len(rows)}행), 평균 halt_step={avg_halt:.1f}")


if __name__ == "__main__":
    main()
