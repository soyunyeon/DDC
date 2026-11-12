#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCC 2026 대학부 본선 - Mission 4 (환자 증상 조기예측) 추론 템플릿
팀원3 end-to-end 미세조정 체크포인트(06_train_gru_e2e.py) 전용 작업 사본.

inference_gru.py(frozen 버전)와의 차이는 Predictor.load_model/step/finalize
뿐이다 - 06_train_gru_e2e.py의 HierE2EModel은 encoder까지 체크포인트에
그대로 저장돼 있으므로, 공개 klue-roberta-base를 다시 받지 않고 체크포인트
state_dict를 전체(encoder 포함) 그대로 로드한다.

실행 예시
---------
    python inference_gru_e2e.py --mission 4-1 \
        --audio_dir ../Work/val_holdout/wav --label_dir ../Work/val_holdout/json \
        --ckpt_path ../Work/ckpt_e2e/best.pt --output ../outputs/mission4-1_e2e_val.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

# =============================================================================
# 고정 상수 (수정 금지)
# =============================================================================

SYMPTOM_CLASSES: List[str] = [
    "고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란",
]
SYMPTOM_SET = set(SYMPTOM_CLASSES)

TARGET_SR = 16000
CSV_SYMPTOM_SEP = ";"


# =============================================================================
# 진행 표시 (수정 금지)
# =============================================================================

def progress(iterable, desc: str = "", total: Optional[int] = None):
    if not sys.stderr.isatty():
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
# 발화조각 / 샘플 정의 (수정 금지)
# =============================================================================

@dataclass
class Utterance:
    index: int
    utt_id: str
    start_ms: int
    end_ms: int
    speaker: int
    text: str
    audio: np.ndarray

    @property
    def duration_sec(self) -> float:
        return len(self.audio) / TARGET_SR


@dataclass
class Sample:
    label_file: str
    audio_path: str
    _raw_utterances: List[dict] = field(repr=False, default_factory=list)
    _waveform: Optional[np.ndarray] = field(repr=False, default=None)

    @property
    def n_utterances(self) -> int:
        return len(self._raw_utterances)

    def stream(self) -> Iterator[Utterance]:
        for i, u in enumerate(self._raw_utterances):
            start_ms = int(u.get("startAt", 0))
            end_ms = int(u.get("endAt", 0))
            yield Utterance(
                index=i,
                utt_id=str(u.get("id", i)),
                start_ms=start_ms,
                end_ms=end_ms,
                speaker=int(u.get("speaker", -1)),
                text=str(u.get("text", "")),
                audio=self._slice(start_ms, end_ms),
            )

    def _slice(self, start_ms: int, end_ms: int) -> np.ndarray:
        if self._waveform is None:
            self._waveform = load_waveform(self.audio_path)
        a = max(0, int(round(start_ms / 1000.0 * TARGET_SR)))
        b = min(len(self._waveform), int(round(end_ms / 1000.0 * TARGET_SR)))
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        return self._waveform[a:b].copy()

    def release(self) -> None:
        self._waveform = None


# =============================================================================
# 오디오 로딩 (수정 금지)
# =============================================================================

def load_waveform(path: str) -> np.ndarray:
    try:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except Exception:
        import librosa
        wav, sr = librosa.load(path, sr=None, mono=True)
        wav = np.asarray(wav, dtype=np.float32)

    if sr != TARGET_SR:
        wav = _resample(wav, sr, TARGET_SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(wav, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        import librosa
        return librosa.resample(wav, orig_sr=sr_in, target_sr=sr_out).astype(np.float32)


# =============================================================================
# 데이터셋 로딩 (수정 금지)
# =============================================================================

def _build_audio_index(audio_dir: str) -> dict:
    index = {}
    for root, _dirs, files in progress(os.walk(audio_dir), "wav 인덱싱"):
        for fn in files:
            if fn.lower().endswith(".wav"):
                index.setdefault(fn, os.path.join(root, fn))
    return index


def _resolve_audio(audio_dir: str, audio_path_field: str, audio_index: dict,
                   label_file: str = "") -> Optional[str]:
    if audio_path_field:
        direct = os.path.join(audio_dir, audio_path_field)
        if os.path.isfile(direct):
            return direct
        hit = audio_index.get(os.path.basename(audio_path_field))
        if hit:
            return hit
    if label_file:
        stem = os.path.splitext(os.path.basename(label_file))[0]
        hit = audio_index.get(stem + ".wav")
        if hit:
            return hit
        direct = os.path.join(audio_dir, stem + ".wav")
        if os.path.isfile(direct):
            return direct
    return None


def load_samples(label_dir: str, audio_dir: str, verbose: bool = True,
                 allow_missing_audio: bool = False) -> List[Sample]:
    audio_index = _build_audio_index(audio_dir)
    json_files = sorted(
        fn for fn in os.listdir(label_dir) if fn.lower().endswith(".json")
    )

    samples: List[Sample] = []
    n_symptom_empty = 0
    n_out_of_class = 0
    n_no_utt = 0
    n_no_audio = 0
    n_by_stem = 0

    for fn in progress(json_files, "라벨 로딩"):
        with open(os.path.join(label_dir, fn), "r", encoding="utf-8") as f:
            d = json.load(f)

        if "symptom" in d:
            raw = d.get("symptom") or []
            if not [s for s in SYMPTOM_CLASSES if s in set(raw)]:
                if raw:
                    n_out_of_class += 1
                else:
                    n_symptom_empty += 1
                continue

        utts = list(d.get("utterances") or [])
        utts.sort(key=lambda u: (int(u.get("startAt", 0)), int(u.get("endAt", 0))))
        if not utts:
            n_no_utt += 1
            continue

        ap_field = d.get("audioPath", "")
        wav_path = _resolve_audio(audio_dir, ap_field, audio_index, label_file=fn)
        if wav_path is None:
            n_no_audio += 1
            continue
        if ap_field and os.path.basename(wav_path) != os.path.basename(ap_field):
            n_by_stem += 1

        samples.append(
            Sample(
                label_file=fn,
                audio_path=wav_path,
                _raw_utterances=utts,
            )
        )

    if verbose:
        print(f"[data] label json           : {len(json_files)}건", file=sys.stderr)
        print(f"[data] symptom 빈 리스트     : {n_symptom_empty}건", file=sys.stderr)
        print(f"[data] 9개 클래스 밖만 보유  : {n_out_of_class}건", file=sys.stderr)
        print(f"[data] 발화 없음             : {n_no_utt}건", file=sys.stderr)
        print(f"[data] wav 미발견 제외       : {n_no_audio}건", file=sys.stderr)
        print(f"[data] (stem 재탐색으로 해결 : {n_by_stem}건)", file=sys.stderr)
        print(f"[data] 평가 대상             : {len(samples)}건", file=sys.stderr)

    if n_no_audio and not allow_missing_audio:
        raise SystemExit(
            f"[error] wav를 찾지 못한 라벨이 {n_no_audio}건 있습니다. "
            f"--audio_dir 경로를 확인하십시오.\n"
            f"        이대로 진행하면 해당 샘플은 채점에서 0점 처리됩니다. "
            f"의도한 것이라면 --allow_missing_audio를 붙이십시오."
        )
    return samples


# =============================================================================
# [참가자 구현 영역] --- 이 클래스만 수정하십시오
#
# end-to-end 미세조정 체크포인트(06_train_gru_e2e.py) 전용.
# HierE2EModel은 encoder/tokenizer를 자기 안에 들고 있고, 체크포인트에
# encoder 가중치까지 포함돼 있으므로 공개 klue-roberta-base를 다시
# 받아올 필요가 없다 - architecture만 만들고 state_dict를 통째로 덮어쓴다.
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("train_gru_e2e_module", _THIS_DIR / "06_train_gru_e2e.py")
_train_gru_e2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_gru_e2e)

HierE2EModel = _train_gru_e2e.HierE2EModel
map_speaker = _train_gru_e2e.map_speaker

from text_clean import clean_for_model

DECISION_THRESHOLD = 0.5    # 주최측 공지: 클래스별 임의 임계값 금지, 0.5 고정


class Predictor:
    def __init__(self, ckpt_path: str, mission: str, device: str = "cpu") -> None:
        self.ckpt_path = ckpt_path
        self.mission = mission
        self.device = device
        self.model = self.load_model(ckpt_path)

    # -------------------------------------------------------------- (1)
    def load_model(self, ckpt_path: str):
        import torch

        ckpt = torch.load(ckpt_path, map_location=self.device)
        config = ckpt["config"]

        model = HierE2EModel(
            backbone=config["backbone"], speaker_dim=config["speaker_dim"],
            hidden=config["hidden"], num_layers=config["num_layers"],
            dropout=config["dropout"], max_length=config["max_length"],
            num_classes=len(config["symptom_classes"]),
        )
        model.load_state_dict(ckpt["model_state"])  # encoder 포함 전체 복원
        model.to(self.device).eval()

        self.symptom_classes = config["symptom_classes"]
        self.hidden = config["hidden"]
        self.num_layers = config["num_layers"]
        self.max_length = config["max_length"]
        self._torch = torch

        return model

    # -------------------------------------------------------------- (2)
    def reset(self) -> None:
        torch = self._torch
        self.h = torch.zeros(self.num_layers, 1, self.hidden, device=self.device)

    # -------------------------------------------------------------- (3)
    def step(self, utt: Utterance) -> Optional[List[str]]:
        torch = self._torch
        model = self.model

        with torch.no_grad():
            text = clean_for_model(utt.text) if utt.text else ""
            inputs = model.tokenizer(
                [text], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            out = model.encoder(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
            emb_vec = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            emb_vec = emb_vec.squeeze(0)

            speaker_id = torch.tensor(map_speaker(utt.speaker), device=self.device)
            spk = model.speaker_emb(speaker_id.view(1, 1))

            # step feature: 06_train_gru_e2e.py의 forward()와 동일하게
            # log1p(지금까지의 발화 순번). utt.index는 "현재 위치"일 뿐
            # 전체 길이 정보가 아니므로 규정상 문제없다.
            step_feat = torch.full((1, 1, 1), float(utt.index), device=self.device).log1p()

            x = torch.cat([emb_vec.view(1, 1, -1), spk, step_feat], dim=-1)
            out_gru, self.h = model.gru(x, self.h)
            logits = model.classifier(out_gru.squeeze(1))
            probs = torch.sigmoid(logits).squeeze(0)

        self.last_probs = probs.tolist()
        pred = [c for c, p in zip(self.symptom_classes, probs.tolist())
                if p > DECISION_THRESHOLD]

        if self.mission == "4-1":
            return pred

        # 4-2: halt 정책 미설계 - 끝까지 듣는 베이스라인 (05_simulate_halt.py로 오프라인 실험)
        return None

    # -------------------------------------------------------------- (4)
    def finalize(self) -> List[str]:
        torch = self._torch
        model = self.model
        with torch.no_grad():
            logits = model.classifier(self.h[-1])
            probs = torch.sigmoid(logits).squeeze(0)
        return [c for c, p in zip(self.symptom_classes, probs.tolist())
                if p > DECISION_THRESHOLD]


# =============================================================================
# 실행 루프 / CSV 저장 (수정 금지)
# =============================================================================

def _normalize(pred: Optional[List[str]]) -> List[str]:
    if pred is None:
        return []
    if isinstance(pred, str):
        pred = [pred]
    got = {str(p).strip() for p in pred}
    return [c for c in SYMPTOM_CLASSES if c in got]


def _fmt(pred: List[str]) -> str:
    return CSV_SYMPTOM_SEP.join(pred)


def run_mission_4_1(predictor: Predictor, samples: List[Sample]):
    rows = []
    prob_rows = []
    for s in progress(samples, "Mission 4-1 추론"):
        predictor.reset()
        for utt in s.stream():
            pred = _normalize(predictor.step(utt))
            rows.append((s.label_file, utt.index + 1, _fmt(pred)))
            prob_rows.append((s.label_file, utt.index + 1, *predictor.last_probs))
        s.release()
    return rows, prob_rows


def write_prob_csv(prob_rows: List[tuple], output: str) -> None:
    header = ["label_file", "step"] + SYMPTOM_CLASSES
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(prob_rows)
    print(f"[done] {output} ({len(prob_rows)}행, 4-2 halt 정책 오프라인 실험용)", file=sys.stderr)


def run_mission_4_2(predictor: Predictor, samples: List[Sample]) -> List[tuple]:
    rows = []
    for s in progress(samples, "Mission 4-2 추론"):
        predictor.reset()
        halted = False
        for utt in s.stream():
            out = predictor.step(utt)
            if out is not None:
                rows.append((s.label_file, utt.index + 1, _fmt(_normalize(out))))
                halted = True
                break
        if not halted:
            rows.append((s.label_file, s.n_utterances, _fmt(_normalize(predictor.finalize()))))
        s.release()
    return rows


def write_csv(rows: List[tuple], mission: str, output: str) -> None:
    header = ["label_file", "step", "symptom"] if mission == "4-1" \
        else ["label_file", "halt_step", "symptom"]
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[done] {output} ({len(rows)}행)", file=sys.stderr)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DCC 2026 대학부 본선 Mission 4 추론 스크립트 (팀원3 GRU e2e)",
    )
    p.add_argument("--mission", required=True, choices=["4-1", "4-2"])
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--allow_missing_audio", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.argv = [sys.argv[0]]

    samples = load_samples(args.label_dir, args.audio_dir,
                           allow_missing_audio=args.allow_missing_audio)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print("[error] 평가 대상 샘플이 없습니다.", file=sys.stderr)
        return 1

    predictor = Predictor(args.ckpt_path, args.mission, device=args.device)

    if args.mission == "4-1":
        rows, prob_rows = run_mission_4_1(predictor, samples)
        write_csv(rows, args.mission, args.output)
        probs_path = os.path.splitext(args.output)[0] + "_probs.csv"
        write_prob_csv(prob_rows, probs_path)
    else:
        rows = run_mission_4_2(predictor, samples)
        write_csv(rows, args.mission, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
