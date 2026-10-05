"""
05c_extract_embeddings_provided_validation.py

03_extract_ecapa_embeddings.py (v5, 동적 배치 버전)과 완전히 동일한 로직을
Work\\ProvidedValidation\\Common\\metadata.csv / caller_segments 에 대해 실행.

같은 frozen ECAPA-TDNN 모델을 그대로 사용 (fine-tuning 없음, inference만).
결과는 Work\\ProvidedValidation\\Common\\embeddings\\shards 에 별도로 저장되어
기존 Work\\Common\\embeddings\\shards (Training 파이프라인 결과물)와 섞이지 않음.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier

try:
    from speechbrain.utils.fetching import LocalStrategy
    _LOCAL_STRATEGY = LocalStrategy.COPY_SKIP_CACHE
except ImportError:
    _LOCAL_STRATEGY = None


BASE_DIR = Path(__file__).resolve().parent.parent

WORK_DIR = BASE_DIR / "Work"
PV_COMMON_DIR = WORK_DIR / "ProvidedValidation" / "Common"

METADATA_PATH = PV_COMMON_DIR / "metadata.csv"
SEGMENT_DIR = PV_COMMON_DIR / "caller_segments"

EMB_DIR = PV_COMMON_DIR / "embeddings"
SHARD_DIR = EMB_DIR / "shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
SHARD_SIZE = 2000  # 전체 규모가 작아서(대략 5~6만 segment 예상) 기존보다 작게
NUM_WORKERS = 4
PREFETCH_FACTOR = 4

MAX_BATCH_AUDIO_SECONDS = 32
BATCH_SIZE_CAP = 32
EMPTY_CACHE_EVERY_N_BATCHES = 20

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_SAVEDIR = BASE_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"


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


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("GPU 사용:" if device == "cuda:0" else "CPU로 진행:", torch.cuda.get_device_name(0) if device == "cuda:0" else "")

    print("ECAPA-TDNN 모델 로드 중...")
    kwargs = dict(source=MODEL_SOURCE, savedir=str(MODEL_SAVEDIR), run_opts={"device": device})
    if _LOCAL_STRATEGY is not None:
        kwargs["local_strategy"] = _LOCAL_STRATEGY

    classifier = EncoderClassifier.from_hparams(**kwargs)
    classifier.eval()

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"metadata.csv가 없습니다: {METADATA_PATH}\n"
            f"05b_preprocess_provided_validation.py를 먼저 실행하세요."
        )

    df = pd.read_csv(METADATA_PATH)
    total_len = len(df)
    print("전체 segment 수:", total_len)

    sorted_df = df.sort_values("duration_ms", kind="mergesort").reset_index(drop=True)
    print(
        "길이 범위:",
        f"{sorted_df['duration_ms'].min():.0f}ms ~ {sorted_df['duration_ms'].max():.0f}ms",
    )

    n_shards = (total_len + SHARD_SIZE - 1) // SHARD_SIZE
    print("shard 개수:", n_shards, f"(shard당 최대 {SHARD_SIZE}개)")

    start_shard = 0
    while start_shard < n_shards:
        shard_npy = SHARD_DIR / f"shard_{start_shard:04d}.npy"
        shard_csv = SHARD_DIR / f"shard_{start_shard:04d}.csv"
        if shard_npy.exists() and shard_csv.exists():
            start_shard += 1
        else:
            break

    if start_shard >= n_shards:
        print("모든 shard가 이미 처리되어 있습니다. 할 일이 없습니다.")
        return

    print(f"{start_shard}개 shard는 이미 완료됨 -> {start_shard}번째 shard부터 이어서 진행")

    remaining_start = start_shard * SHARD_SIZE
    remaining_df = sorted_df.iloc[remaining_start:].reset_index(drop=True)

    dataset = SegmentDataset(remaining_df, SEGMENT_DIR)

    max_batch_samples = int(MAX_BATCH_AUDIO_SECONDS * TARGET_SR)
    batch_sampler = DynamicDurationBatchSampler(
        durations_ms=remaining_df["duration_ms"].tolist(),
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

    ordered_results = [None] * len(remaining_df)
    current_shard_idx = start_shard
    flush_start = 0
    failed_count = 0
    start_time = time.time()
    processed = 0
    batch_counter = 0

    def shard_local_boundary(shard_idx):
        return min((shard_idx - start_shard + 1) * SHARD_SIZE, len(remaining_df))

    boundary = shard_local_boundary(current_shard_idx)

    with torch.inference_mode():
        for idxs, batch_tensor, wav_lens, failed_idxs, valid_idxs in loader:
            if batch_tensor is not None:
                batch_tensor = batch_tensor.to(device, non_blocking=True)
                wav_lens = wav_lens.to(device, non_blocking=True)
                emb = classifier.encode_batch(batch_tensor, wav_lens)
                emb = emb.squeeze(1).cpu().numpy()
                for j, idx in enumerate(valid_idxs):
                    ordered_results[idx] = emb[j]

            failed_count += len(failed_idxs)
            processed += len(idxs)
            batch_counter += 1

            if device == "cuda:0" and batch_counter % EMPTY_CACHE_EVERY_N_BATCHES == 0:
                torch.cuda.empty_cache()

            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(
                f"\r{remaining_start + processed}/{total_len} segment | "
                f"경과 {elapsed / 60:.1f}분 | {rate:.1f}개/초 | 배치 크기 {len(idxs)}",
                end="",
            )

            while processed >= boundary:
                slice_results = ordered_results[flush_start:boundary]
                slice_df = remaining_df.iloc[flush_start:boundary].reset_index(drop=True)
                keep_positions = [i for i, v in enumerate(slice_results) if v is not None]

                if keep_positions:
                    vectors = np.stack([slice_results[i] for i in keep_positions]).astype(np.float32)
                    kept_df = slice_df.iloc[keep_positions].reset_index(drop=True)
                else:
                    vectors = np.zeros((0, 192), dtype=np.float32)
                    kept_df = slice_df.iloc[0:0]

                shard_npy = SHARD_DIR / f"shard_{current_shard_idx:04d}.npy"
                shard_csv = SHARD_DIR / f"shard_{current_shard_idx:04d}.csv"
                np.save(shard_npy, vectors)
                kept_df.to_csv(shard_csv, index=False)

                print(
                    f"\n  -> shard {current_shard_idx:04d} 저장 완료 "
                    f"({len(kept_df)}개, 실패 {len(slice_df) - len(kept_df)}개)"
                )

                flush_start = boundary
                current_shard_idx += 1
                if current_shard_idx >= n_shards:
                    break
                boundary = shard_local_boundary(current_shard_idx)

            if current_shard_idx >= n_shards:
                break

    print()
    print("=" * 60)
    print("Provided-Validation 임베딩 추출 완료")
    print("=" * 60)
    print("총 실패 segment:", failed_count)
    print("저장 위치:", SHARD_DIR)


if __name__ == "__main__":
    main()
