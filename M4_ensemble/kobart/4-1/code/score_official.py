#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCC 2026 대학부 본선 - Mission 4 채점 스크립트 (참가자 공개본)

주최측이 여러분의 제출 CSV를 채점할 때 쓰는 것과 동일한 채점식입니다.
정답 symptom이 들어 있는 라벨 폴더가 있으면 제출 전에 미리 점수를 내볼 수 있습니다.

사용법
------
    python score.py --mission 4-1 --pred ./outputs/mission4-1.csv \
        --label_dir {json folder} --audio_dir {wav folder}
    python score.py --mission 4-2 --pred ./outputs/mission4-2.csv \
        --label_dir {json folder} --audio_dir {wav folder}

--audio_dir를 함께 주면 inference.py와 동일한 기준으로 평가 대상을 확정하므로,
두 스크립트가 처리한 건수가 일치하는지 확인할 수 있습니다.

채점식
------
Mission 4-1  각 (샘플, 순간)마다 클래스별 이진 F1을 구해 평균한다.
             단일 인스턴스이므로 TP는 1.0, FP/FN은 0.0이며,
             정답에도 예측에도 없는 클래스(TN)는 평균에서 제외한다.
             순간 점수를 순간에 대해 평균하여 샘플 점수를 얻고,
             샘플 점수를 샘플에 대해 평균하여 최종 점수로 삼는다.

Mission 4-2  중단 순간 예측에 대한 전체 평가셋 Macro-F1에
             샘플평균 조기판단력을 곱한다.
             조기판단력 = 0.5 + 0.5 x (1 - max(중단까지 입력받은 발화조각 수, 5)
                                         / 전체 발화조각 수)
             - 처음 5개 조각 안에서 중단하면 5번째에서 중단한 것과 같게 본다.
             - 끝까지 듣고 중단해도 최소 0.5를 받는다.

최종 순위    Mission 4-1 순위와 4-2 순위의 합이 작은 팀이 상위.

경고가 출력되면 제출 전에 반드시 해결하십시오. 행 누락, 중복 행, 범위를 벗어난
halt_step, 9개 클래스 밖 문자열, 규격에 맞지 않는 헤더가 대표적입니다.
경고는 화면에 15건까지만 나오므로 전량을 보려면 --out_json을 쓰십시오.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

SYMPTOM_CLASSES: List[str] = [
    "고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란",
]
SYMPTOM_SET = set(SYMPTOM_CLASSES)
CSV_SYMPTOM_SEP = ";"

# Mission 4-2 조기판단력 파라미터
EARLINESS_GRACE = 5       # 이 조각 수 이내의 중단은 모두 같은 조기판단력
EARLINESS_FLOOR = 0.5     # 끝까지 들었을 때 받는 최소 조기판단력


def earliness_weight(halt: int, n_utt: int) -> float:
    """조기판단력 = FLOOR + (1-FLOOR) * (1 - max(halt, GRACE) / n_utt), 범위 [FLOOR, 1)."""
    if n_utt <= 0:
        return EARLINESS_FLOOR
    r = 1.0 - max(halt, EARLINESS_GRACE) / n_utt
    return EARLINESS_FLOOR + (1.0 - EARLINESS_FLOOR) * max(0.0, r)


# =============================================================================
# 진행 표시
#
# tqdm이 있으면 진행바, 없으면 1초마다 stderr 한 줄. tqdm은 선택 사항이다.
# stderr가 터미널이 아니면 아무것도 출력하지 않으므로 채점 로그가 깨끗하다.
# _QUIET은 바깥 루프에 이미 진행바가 있을 때 안쪽 바를 끄는 데 쓴다.
# =============================================================================

_QUIET = False


def set_quiet(v: bool) -> None:
    global _QUIET
    _QUIET = v


def progress(iterable, desc: str = "", total: Optional[int] = None):
    if _QUIET or not sys.stderr.isatty():
        return iterable
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, unit="건",
                    file=sys.stderr, leave=False, dynamic_ncols=True)
    except ImportError:
        return _plain_progress(iterable, desc, total)


def _plain_progress(iterable, desc, total):
    import time
    t0 = last = time.time()
    n = 0
    for x in iterable:
        yield x
        n += 1
        now = time.time()
        if now - last >= 1.0:
            last = now
            tail = f"/{total} ({n / total:.0%})" if total else ""
            print(f"\r  {desc}: {n}{tail}  {now - t0:.0f}s",
                  end="", file=sys.stderr, flush=True)
    if n:
        print(f"\r  {desc}: {n} 완료  {time.time() - t0:.0f}s"
              f"{' ' * 20}", file=sys.stderr)


# =============================================================================
# 정답 로딩
# =============================================================================

class GoldSample:
    __slots__ = ("label_file", "gold", "n_utt", "texts", "speakers")

    def __init__(self, label_file, gold, n_utt, texts, speakers):
        self.label_file = label_file
        self.gold = gold
        self.n_utt = n_utt
        self.texts = texts
        self.speakers = speakers


def load_gold(label_dir: str, verbose: bool = True,
              audio_dir: Optional[str] = None) -> Dict[str, GoldSample]:
    """inference.py의 load_samples와 동일한 규칙으로 평가 대상을 확정한다.

    [FIX D] audio_dir을 주면 inference.py와 똑같이 wav 미발견 샘플도 제외한다.
    주지 않으면 wav를 못 찾아 추론에서 빠진 샘플이 여기서는 평가 대상으로 남아
    조용히 0점 처리되며, 두 스크립트의 평가 대상 건수가 어긋난다.
    """
    audio_index = None
    if audio_dir:
        audio_index = {}
        for root, _dirs, files in progress(os.walk(audio_dir), "wav 인덱싱"):
            for f in files:
                if f.lower().endswith(".wav"):
                    audio_index.setdefault(f, os.path.join(root, f))

    gold: Dict[str, GoldSample] = {}
    n_files = 0
    n_excluded = 0
    n_no_audio = 0

    for fn in progress(sorted(os.listdir(label_dir)), "정답 로딩"):
        if not fn.lower().endswith(".json"):
            continue
        n_files += 1
        with open(os.path.join(label_dir, fn), "r", encoding="utf-8") as f:
            d = json.load(f)

        raw = set(d.get("symptom") or [])
        g = [c for c in SYMPTOM_CLASSES if c in raw]
        if not g:
            n_excluded += 1
            continue

        utts = list(d.get("utterances") or [])
        if not utts:
            n_excluded += 1
            continue
        utts.sort(key=lambda u: (int(u.get("startAt", 0)), int(u.get("endAt", 0))))

        if audio_index is not None:      # [FIX D] inference.py와 동일한 wav 해석
            ap = d.get("audioPath", "")
            stem = os.path.splitext(fn)[0]
            found = (bool(ap) and os.path.isfile(os.path.join(audio_dir, ap))) \
                or (bool(ap) and os.path.basename(ap) in audio_index) \
                or (stem + ".wav" in audio_index) \
                or os.path.isfile(os.path.join(audio_dir, stem + ".wav"))
            if not found:
                n_no_audio += 1
                n_excluded += 1
                continue

        gold[fn] = GoldSample(
            label_file=fn,
            gold=g,
            n_utt=len(utts),
            texts=[str(u.get("text", "")) for u in utts],
            speakers=[int(u.get("speaker", -1)) for u in utts],
        )

    if verbose:
        print(f"[gold] json {n_files}건 중 평가 대상 {len(gold)}건 "
              f"(제외 {n_excluded}건" +
              (f", 그중 wav 미발견 {n_no_audio}건" if audio_dir else "") + ")",
              file=sys.stderr)
    return gold


# =============================================================================
# 제출 CSV 파싱
# =============================================================================

def _parse_symptoms(cell: str, sink: Optional[set] = None) -> List[str]:
    if cell is None:
        return []
    got = {s.strip() for s in str(cell).split(CSV_SYMPTOM_SEP) if s.strip()}
    unknown = got - SYMPTOM_SET
    if unknown:
        got = got & SYMPTOM_SET
        if sink is not None:      # [FIX B] 조용히 버리지 않고 경고로 노출
            sink.update(unknown)
    return [c for c in SYMPTOM_CLASSES if c in got]


def read_pred_csv(path: str, mission: str) -> Tuple[dict, List[str]]:
    """제출 CSV를 읽어 예측 dict과 경고 목록을 반환한다.

    4-1 -> {label_file: {step: [symptom]}}
    4-2 -> {label_file: (halt_step, [symptom])}
    """
    warnings: List[str] = []
    step_col = "step" if mission == "4-1" else "halt_step"

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        if "label_file" not in cols or step_col not in cols or "symptom" not in cols:
            raise ValueError(
                f"CSV 헤더가 규격과 다릅니다. 기대: label_file,{step_col},symptom / 실제: {cols}"
            )

        out: dict = {}
        unknown_labels: set = set()          # [FIX B]
        for ln, row in progress(enumerate(reader, start=2), "제출 CSV 파싱"):
            lf = (row.get("label_file") or "").strip()
            if not lf:
                warnings.append(f"{ln}행: label_file 비어 있음")
                continue
            try:
                step = int(str(row.get(step_col)).strip())
            except (TypeError, ValueError):
                warnings.append(f"{ln}행: {step_col} 파싱 실패")
                continue
            pred = _parse_symptoms(row.get("symptom"), unknown_labels)

            if mission == "4-1":
                if step in out.get(lf, {}):
                    # [FIX C] 같은 (label_file, step) 중복. 기존에는 조용히 덮어썼다.
                    warnings.append(f"{ln}행: {lf} step {step} 중복, 마지막 행 사용")
                out.setdefault(lf, {})[step] = pred
            else:
                if lf in out:
                    warnings.append(f"{ln}행: {lf} 중복 제출, 첫 행 사용")
                    continue
                out[lf] = (step, pred)

    if unknown_labels:      # [FIX B]
        warnings.append(
            f"9개 클래스 밖 문자열 {len(unknown_labels)}종을 무시했습니다: "
            f"{sorted(unknown_labels)[:8]}"
        )
    return out, warnings


# =============================================================================
# Mission 4-1
# =============================================================================

def moment_score(gold: List[str], pred: List[str]) -> float:
    """한 순간의 점수. TN 클래스는 평균에서 제외한다."""
    gs, ps = set(gold), set(pred)
    scored = 0
    total = 0.0
    for c in SYMPTOM_CLASSES:
        g, p = c in gs, c in ps
        if not g and not p:
            continue          # TN -> 평균에서 제외
        scored += 1
        total += 1.0 if (g and p) else 0.0
    if scored == 0:
        return 1.0            # 정답도 예측도 없음 (정답 필터링상 실제로는 발생하지 않음)
    return total / scored


def score_mission_4_1(pred: dict, gold: Dict[str, GoldSample]) -> dict:
    warnings: List[str] = []
    per_sample: Dict[str, float] = {}

    for lf, gs in progress(list(gold.items()), "4-1 채점"):
        steps = pred.get(lf, {})
        if not steps:
            warnings.append(f"{lf}: 예측 없음 -> 0점 처리")
            per_sample[lf] = 0.0
            continue

        missing = 0
        acc = 0.0
        for t in range(1, gs.n_utt + 1):
            if t in steps:
                acc += moment_score(gs.gold, steps[t])
            else:
                missing += 1     # 누락 순간은 0점
        if missing:
            warnings.append(f"{lf}: {gs.n_utt}개 순간 중 {missing}개 누락 -> 해당 순간 0점")
        per_sample[lf] = acc / gs.n_utt

    extra = set(pred) - set(gold)
    if extra:
        warnings.append(f"평가 대상이 아닌 {len(extra)}개 파일의 예측은 무시했습니다.")

    score = sum(per_sample.values()) / len(per_sample) if per_sample else 0.0
    return {"score": score, "per_sample": per_sample, "warnings": warnings}


# =============================================================================
# Mission 4-2
# =============================================================================

def macro_f1(pairs: List[Tuple[List[str], List[str]]]) -> Tuple[float, Dict[str, Optional[float]]]:
    """전체 평가셋을 풀링한 클래스별 F1과 그 매크로 평균.

    정답에도 예측에도 한 번도 등장하지 않은 클래스는 매크로 평균에서 제외한다.
    """
    per_class: Dict[str, Optional[float]] = {}
    vals: List[float] = []
    for c in SYMPTOM_CLASSES:
        tp = fp = fn = 0
        for g, p in pairs:
            gi, pi = c in set(g), c in set(p)
            if gi and pi:
                tp += 1
            elif pi and not gi:
                fp += 1
            elif gi and not pi:
                fn += 1
        if tp + fp + fn == 0:
            per_class[c] = None      # 전 구간 TN -> 제외
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = f1
        vals.append(f1)
    return (sum(vals) / len(vals) if vals else 0.0), per_class


def score_mission_4_2(pred: dict, gold: Dict[str, GoldSample]) -> dict:
    warnings: List[str] = []
    pairs: List[Tuple[List[str], List[str]]] = []
    earliness: List[float] = []
    per_sample: Dict[str, dict] = {}

    for lf, gs in progress(list(gold.items()), "4-2 채점"):
        if lf not in pred:
            warnings.append(f"{lf}: 예측 없음 -> 빈 예측 / 조기판단력 0")
            pairs.append((gs.gold, []))
            earliness.append(0.0)
            per_sample[lf] = {"halt": gs.n_utt, "n_utt": gs.n_utt,
                              "earliness": 0.0, "pred": []}
            continue

        halt, p = pred[lf]
        if halt < 1 or halt > gs.n_utt:
            warnings.append(
                f"{lf}: halt_step {halt}이 범위(1~{gs.n_utt}) 밖 -> {gs.n_utt}로 보정"
            )
            halt = gs.n_utt
        e = earliness_weight(halt, gs.n_utt)
        pairs.append((gs.gold, p))
        earliness.append(e)
        per_sample[lf] = {"halt": halt, "n_utt": gs.n_utt, "earliness": e, "pred": p}

    extra = set(pred) - set(gold)
    if extra:
        warnings.append(f"평가 대상이 아닌 {len(extra)}개 파일의 예측은 무시했습니다.")

    mf1, per_class = macro_f1(pairs)
    mean_e = sum(earliness) / len(earliness) if earliness else 0.0
    return {
        "score": mf1 * mean_e,
        "macro_f1": mf1,
        "mean_earliness": mean_e,
        "per_class_f1": per_class,
        "per_sample": per_sample,
        "warnings": warnings,
    }


# =============================================================================
# 출력
# =============================================================================

def _print_warnings(ws: List[str], limit: int = 15) -> None:
    if not ws:
        return
    print(f"\n[경고] {len(ws)}건", file=sys.stderr)
    for w in ws[:limit]:
        print(f"  - {w}", file=sys.stderr)
    if len(ws) > limit:
        print(f"  ... 외 {len(ws) - limit}건", file=sys.stderr)


def score_single(mission: str, pred_path: str,
                 gold: Dict[str, GoldSample]) -> dict:
    pred, parse_warnings = read_pred_csv(pred_path, mission)
    if mission == "4-1":
        res = score_mission_4_1(pred, gold)
    else:
        res = score_mission_4_2(pred, gold)
    res["warnings"] = parse_warnings + res["warnings"]
    res["_pred"] = pred
    return res


def empty_result(mission: str, reason: str) -> dict:
    """읽을 수 없는 제출물을 0점 + 경고로 처리하기 위한 빈 결과."""
    res = {"score": 0.0, "per_sample": {}, "_pred": {},
           "warnings": [f"제출 파일을 읽을 수 없습니다: {reason} -> 0점 처리"]}
    if mission == "4-2":
        res.update({"macro_f1": 0.0, "mean_earliness": 0.0,
                    "per_class_f1": {c: None for c in SYMPTOM_CLASSES}})
    return res


def print_result(mission: str, res: dict) -> None:
    print(f"\n=== Mission {mission} ===")
    print(f"Score : {res['score']:.4f}")
    if mission == "4-2":
        print(f"  Macro-F1      : {res['macro_f1']:.4f}")
        print(f"  평균 조기판단력 : {res['mean_earliness']:.4f}")
        print("  클래스별 F1:")
        for c, v in res["per_class_f1"].items():
            print(f"    {c:<6} {'제외(미등장)' if v is None else f'{v:.4f}'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DCC 2026 대학부 본선 Mission 4 채점")
    p.add_argument("--label_dir", required=True, help="정답 라벨링 json 폴더")
    p.add_argument("--audio_dir",
                   help="주면 inference.py와 동일하게 wav 미발견 샘플도 평가에서 제외")
    p.add_argument("--mission", choices=["4-1", "4-2"], help="채점할 미션")
    p.add_argument("--pred", help="채점할 csv 경로")
    p.add_argument("--out_json", help="결과를 json으로 저장")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    gold = load_gold(args.label_dir, audio_dir=args.audio_dir)
    if not gold:
        print("[error] 평가 대상 정답이 없습니다.", file=sys.stderr)
        return 1
    if not (args.mission and args.pred):
        print("[error] --mission 과 --pred 를 함께 지정하십시오.", file=sys.stderr)
        return 1

    try:
        res = score_single(args.mission, args.pred, gold)
    except (ValueError, OSError) as e:
        res = empty_result(args.mission, str(e))

    print_result(args.mission, res)
    _print_warnings(res["warnings"])

    if args.out_json:
        res.pop("_pred", None)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
