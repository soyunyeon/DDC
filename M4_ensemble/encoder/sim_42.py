# -*- coding: utf-8 -*-
r"""
sim_42.py : 저장된 4-1 확률 파일로 Mission 4-2 중단 규칙을 시뮬레이션하고 고른다 (재학습, 재추론 없음)

  DCC 본선 폴더에서 (score.py와 같은 폴더):
  py -3.11 sim_42.py --team Data\Work\Encoder\klue_base\val_probs.oof.npz --official Data\Work\Encoder\klue_base\officialval_probs.oof.npz

규칙 (운영사무국 안내 기준)
  - 중단 판단 기준값은 0.5: 어떤 증상이든 확률이 0.5 이상일 때만 중단할 수 있다
  - 중단할 때 내는 답 = 그 시점에서 확률 0.5 이상인 증상 전부 (판정 기준도 0.5)
  - 끝까지 중단하지 않으면 마지막 시점의 0.5 이상 증상을 낸다 (finalize, 비어 있을 수 있음)
  - 쓰는 정보는 지금까지 받은 발화 수 t와 그때까지의 확률뿐. 전체 길이는 채점에만 쓴다
  - 하이퍼파라미터(t_min, k)는 팀 val에서만 고른다. 공식 Validation은 평가만 한다

중단 규칙 (두 하이퍼파라미터)
  t >= t_min 이고, 최근 k개 시점 연속으로 "0.5 이상 증상이 1개 이상 + 그 증상 집합이 같음"이면 중단
  - t_min: 5보다 일찍 멈춰도 조기판단력이 5와 같으므로 5부터 본다
  - k    : 예측이 몇 번 연속 같아야 믿고 멈출지

채점: 공식 score.py의 macro_f1, earliness_weight를 그대로 쓴다 (성능 = Macro-F1 x 평균 조기판단력)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import SYMPTOM_CLASSES, earliness_weight, macro_f1  # noqa: E402  공식 채점 함수

THR = 0.5
T_MINS = (5, 6, 7, 8, 10, 12, 15, 20)
KS = (1, 2, 3, 4)


def load(path):
    z = np.load(path)
    n = sum(k.startswith("arr_") for k in z.files)
    golds = [[g for g in str(x).split(";") if g] for x in z["golds"]]
    return [z[f"arr_{i}"].astype(np.float64) for i in range(n)], golds[:n]


def precompute(seqs):
    """통화마다 시점별 (예측 집합 id, 같은 집합 연속 길이) 계산. 집합 id = 9비트 정수, 0 = 빈 예측"""
    out = []
    bits = 1 << np.arange(len(SYMPTOM_CLASSES))
    for P in seqs:
        sid = ((P >= THR) * bits).sum(1)
        run = np.ones(len(sid), dtype=int)
        for t in range(1, len(sid)):
            if sid[t] == sid[t - 1]:
                run[t] = run[t - 1] + 1
        run[sid == 0] = 0                       # 빈 예측은 확신 없음
        out.append((sid, run))
    return out


def to_labels(s):
    return [c for i, c in enumerate(SYMPTOM_CLASSES) if (s >> i) & 1]


def simulate(pre, golds, t_min, k):
    """정책 하나로 모든 통화를 돌려 공식 4-2 점수를 낸다. t_min=None이면 끝까지 듣기."""
    pairs, early, halts = [], [], []
    for (sid, run), g in zip(pre, golds):
        n = len(sid)
        h = n                                   # 기본: 끝까지 (finalize)
        if t_min is not None:
            ok = np.nonzero(run >= k)[0]
            ok = ok[ok + 1 >= t_min]            # t = index + 1
            if len(ok):
                h = int(ok[0]) + 1
        pairs.append((g, to_labels(sid[h - 1])))
        early.append(earliness_weight(h, n))
        halts.append(h / n)
    mf1, per_class = macro_f1(pairs)
    me = float(np.mean(early))
    return dict(score=mf1 * me, mf1=mf1, early=me, halt_ratio=float(np.mean(halts)),
                halted=float(np.mean([r < 1 for r in halts])), per_class=per_class)


def fmt(r):
    return (f"4-2 {r['score']:.4f} | Macro-F1 {r['mf1']:.4f} x 조기판단력 {r['early']:.4f} | "
            f"평균 중단 위치 {r['halt_ratio']:.0%} | 끝 전에 중단 {r['halted']:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="팀 val 확률 파일 (정책 선택용)")
    ap.add_argument("--official", default=None, help="같은 모델의 공식 Validation 확률 파일 (평가만)")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    st, gt = load(a.team)
    pt = precompute(st)
    print(f"[team] {a.team} : 통화 {len(st)}건")
    grid = [(tm, k) for tm in T_MINS for k in KS]
    rows = sorted(((simulate(pt, gt, tm, k), tm, k) for tm, k in grid), key=lambda x: -x[0]["score"])
    ref = simulate(pt, gt, None, 1)

    po = go = None
    if a.official:
        so, go = load(a.official)
        po = precompute(so)
        print(f"[official] {a.official} : 통화 {len(so)}건 (평가만)")

    print(f"\n[참고] 끝까지 듣기       팀 val {fmt(ref)}")
    print(f"\n[팀 val 상위 {a.top}개 정책] (공식 Validation 점수는 같은 정책을 그대로 적용한 결과)")
    for r, tm, k in rows[:a.top]:
        line = f"  t_min {tm:>2}, k {k} | 팀 val {r['score']:.4f} (F1 {r['mf1']:.4f} x {r['early']:.4f})"
        if po is not None:
            o = simulate(po, go, tm, k)
            line += f" | 공식 {o['score']:.4f} (F1 {o['mf1']:.4f} x {o['early']:.4f})"
        print(line)

    best, tm, k = rows[0]
    print(f"\n[선택] t_min {tm}, k {k} (팀 val 1위)")
    print(f"  팀 val   {fmt(best)}")
    if po is not None:
        o = simulate(po, go, tm, k)
        print(f"  공식 val {fmt(o)}")
        print(f"  공식 val 끝까지 듣기 {fmt(simulate(po, go, None, 1))}")
        print("  클래스별 F1 (공식 val): " + ", ".join(
            f"{c} {v:.2f}" if v is not None else f"{c} -" for c, v in o["per_class"].items()))


if __name__ == "__main__":
    main()