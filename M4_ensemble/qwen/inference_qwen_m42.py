#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCC 2026 대학부 본선 - Mission 4 (환자 증상 조기예측) 추론 템플릿

실행 예시
---------
    python inference.py --mission 4-1 \
        --audio_dir ./data/wav --label_dir ./data/label \
        --ckpt_path ./ckpt/model.pt --output ./outputs/mission4-1.csv

    python inference.py --mission 4-2 \
        --audio_dir ./data/wav --label_dir ./data/label \
        --ckpt_path ./ckpt/model.pt --output ./outputs/mission4-2.csv

참가자가 작성할 부분은 아래 [참가자 구현 영역] 의 Predictor 클래스뿐입니다.
그 밖의 코드(데이터 로딩, 발화조각 오디오 절단, 스트리밍, CSV 저장)는
채점 재현성을 위해 수정하지 마십시오.

핵심 제약
---------
* 대화의 전체 길이(총 발화조각 수, 총 통화 시간)는 입력으로 사용할 수 없습니다.
  본 템플릿은 발화조각을 한 개씩만 흘려보내므로 참가자 코드가 전체 길이를
  알 수 있는 경로 자체가 없습니다. 우회 시도(원본 wav 재로딩 등)는 실격입니다.
* symptom 라벨은 아래 9개 클래스만 취급합니다.
* 9개 클래스로 필터링한 뒤 symptom이 빈 리스트가 되는 샘플은 평가에서 제외됩니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

# =============================================================================
# 고정 상수 (수정 금지)
# =============================================================================

SYMPTOM_CLASSES: List[str] = [
    "고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란",
]
SYMPTOM_SET = set(SYMPTOM_CLASSES)

TARGET_SR = 16000          # 모든 발화조각은 16kHz 모노 float32로 통일하여 제공됩니다.
CSV_SYMPTOM_SEP = ";"      # 다중 라벨 구분자


# =============================================================================
# 진행 표시 (수정 금지)
#
# tqdm이 설치돼 있으면 진행바를, 없으면 1초마다 stderr에 한 줄을 찍습니다.
# tqdm은 선택 사항이며 없어도 동작에 지장이 없습니다.
# stderr가 터미널이 아니면(로그 리다이렉션 등) 아무것도 출력하지 않습니다.
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
    """참가자 코드에 전달되는 발화조각 1개.

    Attributes
    ----------
    index : int
        대화 내 0-based 발화 순번. 총 개수는 알 수 없습니다.
    utt_id : str
        라벨링데이터의 발화 id.
    start_ms, end_ms : int
        통화 시작 기준 발화 구간(밀리초).
    speaker : int
        0 = 119대원, 1 = 신고자.
    text : str
        전사 텍스트.
    audio : np.ndarray
        해당 구간의 파형. shape (n_samples,), dtype float32, 16kHz 모노.
    """

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
    """한 건의 119 신고 통화. 참가자 코드에는 전달되지 않습니다.

    [FIX 1] 정답(symptom)은 이 객체에 두지 않습니다. 참가자 코드가
    sys._getframe()으로 호출스택을 거슬러 오르거나 gc.get_objects()로
    이 인스턴스에 도달하면 정답이 그대로 노출되기 때문입니다.
    정답은 load_samples()의 지역변수로만 쓰이고 적격 판정 직후 버려집니다.
    채점은 score.py가 라벨 폴더에서 직접 읽습니다.
    """

    label_file: str
    audio_path: str
    _raw_utterances: List[dict] = field(repr=False, default_factory=list)
    _waveform: Optional[np.ndarray] = field(repr=False, default=None)

    @property
    def n_utterances(self) -> int:
        """채점·CSV 작성용. 참가자 코드에 노출되지 않습니다."""
        return len(self._raw_utterances)

    def stream(self) -> Iterator[Utterance]:
        """발화조각을 시간순으로 한 개씩 생성합니다."""
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
    """wav를 16kHz 모노 float32 numpy 배열로 로드합니다."""
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
    """라벨링데이터의 audioPath를 실제 wav 경로로 해석합니다.

    [FIX 2] 배포되는 원천데이터의 wav 파일명은 라벨 json과 같은 stem
    ({_id}_{날짜}.wav)이고, audioPath가 가리키는 AI-hub 내부 경로
    (20230925/Seoul/.../converted_[...].wav)와 일치하지 않습니다.
    따라서 3순위로 라벨 파일명 stem을 사용한 재탐색을 추가합니다.
    """
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
    """라벨 폴더의 JSON을 읽어 평가 대상 샘플 목록을 만듭니다.

    * 발화조각은 startAt 기준 오름차순으로 정렬합니다.
    * symptom은 9개 클래스만 남기고, 필터 후 빈 리스트가 되면 샘플을 제외합니다.
      (배포되는 평가셋에는 symptom 키가 아예 없으므로 이 필터는 참가자가
       자기 검증셋으로 돌릴 때만 동작합니다.)
    """
    audio_index = _build_audio_index(audio_dir)
    json_files = sorted(
        fn for fn in os.listdir(label_dir) if fn.lower().endswith(".json")
    )

    samples: List[Sample] = []
    n_symptom_empty = 0        # symptom이 원래 빈 리스트
    n_out_of_class = 0         # 9개 클래스 밖 라벨만 보유
    n_no_utt = 0               # [FIX 3] 기존 코드가 집계하지 않던 사유
    n_no_audio = 0
    n_by_stem = 0              # [FIX 2] stem 재탐색으로 찾은 건수

    for fn in progress(json_files, "라벨 로딩"):
        with open(os.path.join(label_dir, fn), "r", encoding="utf-8") as f:
            d = json.load(f)

        # [FIX 1] gold는 적격 판정에만 쓰는 지역변수다. 어디에도 저장하지 않는다.
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

    # [FIX 3] wav를 못 찾은 건은 채점 시 조용히 0점이 됩니다. score.py는 wav
    # 존재 여부를 모르므로 두 스크립트의 평가 대상이 어긋납니다. 기본값에서는
    # 즉시 실패시켜 참가자가 경로 문제를 반드시 인지하게 합니다.
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
# =============================================================================

class Predictor:
    """Qwen3-4B QLoRA symptom classifier + M4-2 streaming stop policy.

    Changes are restricted to this Predictor class. The organizer's streaming,
    audio handling, CSV output, and evaluation plumbing remain unchanged.

    --ckpt_path must point to a PEFT adapter directory containing
    adapter_config.json and adapter_model.safetensors.

    The base model is expected under the sibling Work/Models/Qwen3-4B,
    or can be specified with environment variable DCC_QWEN_BASE_MODEL.

    Both missions use only the current and earlier utterances. The evaluator's
    call length, labels, files and other hidden state are never accessed.
    """

    _MAX_LENGTH = 1024
    _SYMPTOM_THRESHOLD = 0.5
    _MIN_STOP_STEP = 5
    _STABLE_COUNT = 2
    _STOP_CONFIDENCE = 0.70
    _MAX_STOP_STEP = 28

    # IMPORTANT: These speaker strings must EXACTLY match the training data's
    # format_utterance() function in 03_build_prefix_dataset.py.
    _CALLER_NAME = "신고자"
    _DISPATCHER_NAME = "119대원"

    def __init__(self, ckpt_path: str, mission: str, device: str = "cpu") -> None:
        self.ckpt_path = ckpt_path
        self.mission = mission
        self.device = device
        self.model = self.load_model(ckpt_path)
        self.reset()

    def load_model(self, ckpt_path: str):
        """Load the original 4-bit Qwen3-4B and the saved v1 PEFT adapter."""
        from pathlib import Path

        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "This Qwen3-4B 4-bit checkpoint requires a CUDA GPU. "
                "Run with --device cuda on a machine with sufficient VRAM."
            )

        adapter_dir = Path(ckpt_path).expanduser().resolve()
        if adapter_dir.is_file() and adapter_dir.name == "adapter_config.json":
            adapter_dir = adapter_dir.parent
        required = ("adapter_config.json", "adapter_model.safetensors")
        for filename in required:
            if not (adapter_dir / filename).is_file():
                raise FileNotFoundError(
                    f"Missing {filename} under adapter folder {adapter_dir}. "
                    "Pass the PEFT adapter directory to --ckpt_path."
                )

        # Layout from existing work: Qwen3_LoRA/{Models,Checkpoints}/...
        import os
        base_override = os.environ.get("DCC_QWEN_BASE_MODEL")
        if base_override:
            base_dir = Path(base_override).expanduser().resolve()
        else:
            candidates = [
                parent / "Models" / "Qwen3-4B"
                for parent in adapter_dir.parents
            ]
            base_dir = next((p for p in candidates if (p / "config.json").is_file()), None)
        if base_dir is None or not (base_dir / "config.json").is_file():
            raise FileNotFoundError(
                "Qwen3-4B base model not found. Set DCC_QWEN_BASE_MODEL "
                "to the local base model directory before running."
            )

        self._torch = torch
        self._device = torch.device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(base_dir), local_files_only=True
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"
        self._tokenizer.truncation_side = "right"

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # transformers 4.53.3 uses torch_dtype, NOT dtype.
        model = AutoModelForSequenceClassification.from_pretrained(
            str(base_dir),
            num_labels=9,
            problem_type="multi_label_classification",
            quantization_config=quant,
            torch_dtype=torch.bfloat16,
            device_map={"": 0 if self._device.index is None else self._device.index},
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model.config.pad_token_id = self._tokenizer.pad_token_id
        model.config.use_cache = False
        model = PeftModel.from_pretrained(
            model, str(adapter_dir), is_trainable=False, local_files_only=True
        )
        model.eval()
        return model

    def reset(self) -> None:
        """Discard all past utterances and stop state between separate calls."""
        self._utterance_lines: List[str] = []
        self._previous_mask: Optional[int] = None
        self._repeat_count = 0
        self._latest_symptoms: List[str] = []
        self._stopped = False

    def _predict_probabilities(self) -> List[float]:
        """Match v1 validation: joined text, right truncation, sigmoid(logits)."""
        text = "\n".join(self._utterance_lines)
        inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=self._MAX_LENGTH,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits[0].float()
            probabilities = self._torch.sigmoid(logits).cpu().tolist()
        if len(probabilities) != len(SYMPTOM_CLASSES):
            raise ValueError("Qwen classifier must return exactly nine logits")
        return probabilities

    @staticmethod
    def _mask_and_symptoms(probabilities):
        mask = sum(
            int(float(value) >= Predictor._SYMPTOM_THRESHOLD) << i
            for i, value in enumerate(probabilities)
        )
        names = [
            symptom
            for i, symptom in enumerate(SYMPTOM_CLASSES)
            if (mask >> i) & 1
        ]
        return mask, names

    def step(self, utt: Utterance) -> Optional[List[str]]:
        """Process exactly one new utterance; no lookahead or total length."""
        if self._stopped:
            raise RuntimeError("Prediction already halted. Call reset() for a new call.")

        who = self._CALLER_NAME if int(utt.speaker) == 1 else self._DISPATCHER_NAME
        self._utterance_lines.append(f"{who}: {utt.text}")
        probabilities = self._predict_probabilities()
        mask, predicted = self._mask_and_symptoms(probabilities)
        self._latest_symptoms = predicted

        if self.mission == "4-1":
            return predicted
        if self.mission != "4-2":
            raise ValueError(f"Unknown mission: {self.mission}")

        # Offline-vetted policy k2_c70_cap28; only present/past probabilities.
        self._repeat_count = (
            self._repeat_count + 1 if mask == self._previous_mask else 1
        )
        self._previous_mask = mask
        steps = len(self._utterance_lines)
        stable_and_confident = (
            steps >= self._MIN_STOP_STEP
            and mask != 0
            and self._repeat_count >= self._STABLE_COUNT
            and max(probabilities) >= self._STOP_CONFIDENCE
        )
        if stable_and_confident or steps >= self._MAX_STOP_STEP:
            self._stopped = True
            return predicted
        return None

    def finalize(self) -> List[str]:
        """If stream naturally ends, use the latest computed prediction."""
        self._stopped = True
        return self._latest_symptoms.copy()


# =============================================================================
# 실행 루프 / CSV 저장 (수정 금지)
# =============================================================================

def _normalize(pred: Optional[List[str]]) -> List[str]:
    """예측을 9개 클래스로 필터링하고 정해진 순서로 정렬·중복 제거합니다."""
    if pred is None:
        return []
    if isinstance(pred, str):
        pred = [pred]
    got = {str(p).strip() for p in pred}
    return [c for c in SYMPTOM_CLASSES if c in got]


def _fmt(pred: List[str]) -> str:
    return CSV_SYMPTOM_SEP.join(pred)


def run_mission_4_1(predictor: Predictor, samples: List[Sample]) -> List[tuple]:
    rows = []
    for s in progress(samples, "Mission 4-1 추론"):
        predictor.reset()
        for utt in s.stream():
            pred = _normalize(predictor.step(utt))
            rows.append((s.label_file, utt.index + 1, _fmt(pred)))
        s.release()
    return rows


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
        description="DCC 2026 대학부 본선 Mission 4 추론 스크립트",
    )
    p.add_argument("--mission", required=True, choices=["4-1", "4-2"],
                   help="4-1: 누적입력 증상 분류 / 4-2: 중단 시스템")
    p.add_argument("--audio_dir", required=True, help="wav 파일 루트 폴더")
    p.add_argument("--label_dir", required=True, help="라벨링 json 폴더")
    p.add_argument("--ckpt_path", required=True, help="학습 완료된 가중치 파일")
    p.add_argument("--output", required=True, help="결과 csv 경로")
    p.add_argument("--device", default="cpu", help="cpu 또는 cuda")
    p.add_argument("--limit", type=int, default=0,
                   help="디버깅용. 앞 N건만 처리 (0이면 전체)")
    p.add_argument("--allow_missing_audio", action="store_true",
                   help="wav를 찾지 못한 라벨이 있어도 계속 진행 (해당 샘플 0점)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # [FIX 4] 참가자 코드가 sys.argv에서 --label_dir / --audio_dir를 읽어
    # 원본 wav와 정답 라벨에 직접 접근하는 우회를 차단합니다.
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
        rows = run_mission_4_1(predictor, samples)
    else:
        rows = run_mission_4_2(predictor, samples)

    write_csv(rows, args.mission, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
