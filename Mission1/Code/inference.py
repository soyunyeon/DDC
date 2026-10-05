"""
inference.py

Mission 1 (신고자 성별 분류) 최종 제출용 추론 스크립트.

사용법 (대회 규정 11페이지와 동일):
    python inference.py --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path {checkpoint file} --output ./outputs/mission1.csv

동작 순서:
    1. label_dir의 JSON에서 speaker==1(신고자) 구간(startAt, endAt)만 사용해 crop
       -> JSON의 "gender" 필드는 이 스크립트 어디에서도 읽지 않음
          (Mission 1 규정: "inference시 startAt, endAt, Speaker만 사용 가능")
    2. 각 구간을 16kHz mono로 리샘플링 후, frozen ECAPA-TDNN
       (speechbrain/spkrec-ecapa-voxceleb)으로 192차원 임베딩 추출
       (VRAM 안전을 위해 duration 기준 동적 배치 사용, RTX 3050 4GB에서 검증됨)
    3. ckpt_path에 저장된 분류기(단일 모델 또는 여러 seed 앙상블)로
       각 segment의 "여성일 확률"을 예측 후 평균
    4. 통화(call) 단위로 duration 가중평균 집계 (긴 구간에 더 높은 가중치)
    5. threshold 적용해서 최종 M/F 결정
    6. [audio file name, gender] 형식으로 output csv 저장

ckpt_path에 기대하는 내용물 (04_final_train_ensemble.py가 torch.save로 생성):
    {
        "scaler": StandardScaler,
        "clfs": [분류기, ...] 또는 "clf": 단일분류기,
        "preproc_kind": "raw" 또는 "l2",
        "weight_mode": "mean" 또는 "duration",
        "duration_weight_cap_ms": float (weight_mode="duration"일 때만),
        "threshold": float,
        "calibrator": Platt scaling용 LogisticRegression 또는 None,
        "use_calibration": bool (True면 calibrator를 통화 단위 확률에 적용),
    }

Probability Calibration: ckpt에 calibrator가 있고 use_calibration=True이면, 통화
단위로 집계한 확률에 Platt scaling을 적용한 뒤 threshold(0.5)와 비교. 대회 규정
Q4에 따라 확률값 자체를 보정하는 것일 뿐이며, Decision Threshold는 여전히 0.5로
고정(04_final_train_ensemble.py와 동일한 to_logit/apply_calibration 로직).
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import normalize

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier

try:
    from speechbrain.utils.fetching import LocalStrategy
    _LOCAL_STRATEGY = LocalStrategy.COPY_SKIP_CACHE
except ImportError:
    _LOCAL_STRATEGY = None


CALLER_SPEAKER = 1
TARGET_SR = 16000

MAX_BATCH_AUDIO_SECONDS = 32
BATCH_SIZE_CAP = 32
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
EMPTY_CACHE_EVERY_N_BATCHES = 20  # VRAM 파편화 방지 (03_extract_ecapa_embeddings.py v5와 동일)

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


# ---- 인자 파싱 ----

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True, help="wav 폴더 (원천데이터)")
    p.add_argument("--label_dir", required=True, help="json 폴더 (라벨링데이터)")
    p.add_argument("--ckpt_path", required=True, help="학습된 체크포인트 파일 (.ckpt)")
    p.add_argument("--output", required=True, help="결과 csv 저장 경로")
    return p.parse_args()


# ---- 파일 인덱싱 / 신고자 구간 crop (gender 필드는 전혀 읽지 않음) ----

def build_index(root, ext):
    root = Path(root)
    index = {}
    for path in root.rglob(f"*{ext}"):
        index[path.stem] = path
    return index


def crop_caller_segments(json_path, wav_path, stem, out_dir):
    """speaker==1(신고자) 구간만 crop해서 out_dir에 저장.
    startAt, endAt, speaker 외 다른 필드는 읽지 않음 (gender 미사용)."""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    utterances = [
        u for u in data.get("utterances", [])
        if u.get("speaker") == CALLER_SPEAKER
    ]
    utterances.sort(key=lambda x: x.get("startAt", 0))

    y, sr = sf.read(wav_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    audio_length = len(y)

    rows = []
    seg_id = 0

    for utt in utterances:
        start_ms = utt.get("startAt")
        end_ms = utt.get("endAt")

        if start_ms is None or end_ms is None or end_ms <= start_ms:
            continue

        start_sample = round(start_ms / 1000.0 * sr)
        end_sample = round(end_ms / 1000.0 * sr)
        start_sample = max(0, min(start_sample, audio_length))
        end_sample = max(0, min(end_sample, audio_length))

        if end_sample <= start_sample:
            continue

        clip = y[start_sample:end_sample]
        out_name = f"{stem}_seg{seg_id:03d}.wav"
        out_path = out_dir / out_name
        sf.write(out_path, clip, sr)

        duration_ms = len(clip) / sr * 1000.0
        rows.append({
            "call_id": stem,
            "segment_id": seg_id,
            "segment_file": out_name,
            "duration_ms": round(duration_ms, 3),
        })
        seg_id += 1

    return rows


# ---- 임베딩 추출용 Dataset / collate_fn / 동적 배치 샘플러 ----
# (03_extract_ecapa_embeddings.py v5와 동일 로직, RTX 3050 4GB에서 검증됨)

def to_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def apply_calibration(calibrator, call_proba):
    """04_final_train_ensemble.py에서 val_calls로 학습한 Platt scaling을 그대로 적용.
    확률값만 보정하며, Decision Threshold(0.5)는 그대로 유지."""
    return calibrator.predict_proba(to_logit(call_proba).reshape(-1, 1))[:, 1]


def load_and_resample(path_str):
    wav, sr = torchaudio.load(path_str)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)


class SegmentDataset(Dataset):
    def __init__(self, sorted_df, segment_dir):
        self.df = sorted_df.reset_index(drop=True)
        self.segment_dir = segment_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.segment_dir / row["segment_file"]
        try:
            wav = load_and_resample(str(path))
            if wav.shape[0] < 160:
                raise ValueError("too short")
            return idx, wav
        except Exception:
            return idx, None


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[0])
    idxs = [idx for idx, wav in batch]
    valid_pairs = [(idx, wav) for idx, wav in batch if wav is not None]
    failed_idxs = [idx for idx, wav in batch if wav is None]

    if not valid_pairs:
        return idxs, None, None, failed_idxs, []

    valid_idxs, wavs = zip(*valid_pairs)
    lengths = torch.tensor([w.shape[0] for w in wavs])
    max_len = int(lengths.max().item())

    batch_tensor = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        batch_tensor[i, : w.shape[0]] = w

    wav_lens = lengths.float() / max_len
    return idxs, batch_tensor, wav_lens, failed_idxs, list(valid_idxs)


class DynamicDurationBatchSampler(Sampler):
    def __init__(self, durations_ms, sample_rate, max_batch_samples, batch_size_cap):
        self.durations_samples = [
            max(1, int(d / 1000.0 * sample_rate)) for d in durations_ms
        ]
        self.max_batch_samples = max_batch_samples
        self.batch_size_cap = batch_size_cap

    def __iter__(self):
        n = len(self.durations_samples)
        i = 0
        while i < n:
            batch = [i]
            j = i + 1
            while j < n and len(batch) < self.batch_size_cap:
                candidate_len = self.durations_samples[j]
                candidate_count = len(batch) + 1
                if candidate_count * candidate_len > self.max_batch_samples:
                    break
                batch.append(j)
                j += 1
            yield batch
            i = j

    def __len__(self):
        if not self.durations_samples:
            return 0
        avg_len = sum(self.durations_samples) / len(self.durations_samples)
        est_batch_size = max(1, min(self.batch_size_cap, int(self.max_batch_samples / max(1, avg_len))))
        return (len(self.durations_samples) + est_batch_size - 1) // est_batch_size


# ---- 메인 ----

def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    label_dir = Path(args.label_dir)
    ckpt_path = Path(args.ckpt_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ---- 체크포인트 로드 ----
    print("체크포인트 로드 중:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    scaler = ckpt["scaler"]
    clfs = ckpt.get("clfs")
    if clfs is None:
        clfs = [ckpt["clf"]]  # 단일 모델 체크포인트도 호환
    preproc_kind = ckpt.get("preproc_kind", "raw")
    weight_mode = ckpt.get("weight_mode", "mean")
    duration_cap = ckpt.get("duration_weight_cap_ms", 3000)
    threshold = ckpt.get("threshold", 0.5)
    calibrator = ckpt.get("calibrator") if ckpt.get("use_calibration", False) else None

    print(f"  분류기 {len(clfs)}개, preproc={preproc_kind}, weight_mode={weight_mode}, threshold={threshold}, "
          f"calibration={'ON' if calibrator is not None else 'OFF'}")

    # ---- ECAPA-TDNN (frozen) 로드 ----
    print("ECAPA-TDNN 모델 로드 중...")
    model_savedir = Path(__file__).resolve().parent.parent / "pretrained_models" / "spkrec-ecapa-voxceleb"
    kwargs = dict(source=MODEL_SOURCE, savedir=str(model_savedir), run_opts={"device": device})
    if _LOCAL_STRATEGY is not None:
        kwargs["local_strategy"] = _LOCAL_STRATEGY
    classifier = EncoderClassifier.from_hparams(**kwargs)
    classifier.eval()

    # ---- 오디오/라벨 매칭 ----
    wav_index = build_index(audio_dir, ".wav")
    json_index = build_index(label_dir, ".json")
    matched_stems = sorted(set(wav_index) & set(json_index))
    print(f"오디오 {len(wav_index)}개, 라벨 {len(json_index)}개, 매칭 {len(matched_stems)}개")

    missing_json = set(wav_index) - set(json_index)
    if missing_json:
        print(f"[경고] 라벨이 없는 오디오 {len(missing_json)}개는 처리에서 제외됩니다.")

    # ---- 임시 폴더에 신고자 구간 crop ----
    tmp_dir = Path(tempfile.mkdtemp(prefix="mission1_segments_"))
    print("임시 segment 폴더:", tmp_dir)

    all_rows = []
    for stem in matched_stems:
        try:
            rows = crop_caller_segments(json_index[stem], wav_index[stem], stem, tmp_dir)
        except Exception as e:
            print(f"[경고] {stem} 처리 중 오류: {e}")
            rows = []
        all_rows.extend(rows)

    meta = pd.DataFrame(all_rows, columns=["call_id", "segment_id", "segment_file", "duration_ms"])
    print("전체 신고자 segment:", len(meta))

    # ---- 임베딩 추출 (duration 기준 동적 배치) ----
    call_proba = pd.Series(dtype=float)

    if len(meta) > 0:
        sorted_meta = meta.sort_values("duration_ms", kind="mergesort").reset_index(drop=True)
        dataset = SegmentDataset(sorted_meta, tmp_dir)

        max_batch_samples = int(MAX_BATCH_AUDIO_SECONDS * TARGET_SR)
        batch_sampler = DynamicDurationBatchSampler(
            durations_ms=sorted_meta["duration_ms"].tolist(),
            sample_rate=TARGET_SR,
            max_batch_samples=max_batch_samples,
            batch_size_cap=BATCH_SIZE_CAP,
        )

        loader_kwargs = dict(
            batch_sampler=batch_sampler,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=(device == "cuda:0"),
        )
        if NUM_WORKERS > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

        loader = DataLoader(dataset, **loader_kwargs)

        embeddings = [None] * len(sorted_meta)
        processed = 0
        batch_counter = 0
        t0 = time.time()

        with torch.inference_mode():
            for idxs, batch_tensor, wav_lens, failed_idxs, valid_idxs in loader:
                if batch_tensor is not None:
                    batch_tensor = batch_tensor.to(device, non_blocking=True)
                    wav_lens = wav_lens.to(device, non_blocking=True)
                    emb = classifier.encode_batch(batch_tensor, wav_lens)
                    emb = emb.squeeze(1).cpu().numpy()
                    for j, idx in enumerate(valid_idxs):
                        embeddings[idx] = emb[j]

                processed += len(idxs)
                batch_counter += 1

                # VRAM 파편화 방지 (이게 빠지면 VRAM이 좁은 GPU에서 극심한 속도저하 발생 - 실측 확인됨)
                if device == "cuda:0" and batch_counter % EMPTY_CACHE_EVERY_N_BATCHES == 0:
                    torch.cuda.empty_cache()

                elapsed = time.time() - t0
                print(f"\r임베딩 추출: {processed}/{len(sorted_meta)} | 경과 {elapsed:.1f}초", end="")

        print()

        valid_mask = [e is not None for e in embeddings]
        n_valid = sum(valid_mask)

        if n_valid > 0:
            X = np.stack([e for e in embeddings if e is not None]).astype(np.float32)
            meta_valid = sorted_meta[valid_mask].reset_index(drop=True)

            # ---- 전처리 + 분류 (앙상블 평균) ----
            Xp = normalize(X, norm="l2") if preproc_kind == "l2" else X
            Xp = scaler.transform(Xp)

            proba_stack = np.stack([clf.predict_proba(Xp)[:, 1] for clf in clfs], axis=0)
            seg_proba = proba_stack.mean(axis=0)

            meta_valid = meta_valid.copy()
            meta_valid["proba"] = seg_proba

            # ---- 통화 단위 집계 ----
            if weight_mode == "duration":
                w = meta_valid["duration_ms"].clip(upper=duration_cap)
                meta_valid["w_proba"] = meta_valid["proba"] * w
                grouped = meta_valid.groupby("call_id").agg(
                    w_sum=("w_proba", "sum"),
                    w_total=("duration_ms", lambda s: s.clip(upper=duration_cap).sum()),
                )
                call_proba = grouped["w_sum"] / grouped["w_total"]
            else:
                call_proba = meta_valid.groupby("call_id")["proba"].mean()

            # ---- 확률 calibration 적용 (ckpt에 있을 때만, 규정 Q4) ----
            if calibrator is not None:
                call_proba = pd.Series(
                    apply_calibration(calibrator, call_proba.to_numpy()),
                    index=call_proba.index,
                )

    # ---- 결과 조립 (오디오/라벨이 매칭된 모든 call에 대해 출력) ----
    results = []
    fallback_used = 0

    for stem in matched_stems:
        if stem in call_proba.index:
            p = call_proba.loc[stem]
            gender = "F" if p >= threshold else "M"
        else:
            # 유효한 신고자 구간이 하나도 없었던 극히 예외적인 경우의 안전장치
            gender = "M"
            fallback_used += 1

        results.append({
            "audio file name": wav_index[stem].name,
            "gender": gender,
        })

    if fallback_used > 0:
        print(f"[경고] 유효 신고자 구간이 없어 기본값(M)으로 처리된 통화: {fallback_used}개")

    result_df = pd.DataFrame(results, columns=["audio file name", "gender"])
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("결과 저장:", output_path, f"({len(result_df)}행)")

    # ---- 임시 폴더 정리 ----
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("임시 폴더 정리 완료.")


if __name__ == "__main__":
    main()
