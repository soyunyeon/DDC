"""
03_extract_ecapa_embeddings.py (v5, VRAM 폭주 문제 수정)

v4까지의 문제:
    duration 순으로 정렬은 했지만, 배치 "개수"가 32개로 고정되어 있었음.
    -> 뒤로 갈수록(오디오가 길어질수록) "32개 x 점점 길어지는 오디오"를
       그대로 GPU에 올리면서 4GB VRAM을 거의 다 채움.
    -> nvidia-smi 확인 결과: Memory-Usage 3934MiB/4096MiB, GPU-Util 100%인데
       Power Usage는 15W/16W (정상 시 35~60W 수준) -> 실제 연산이 아니라
       메모리 할당/해제(파편화) 오버헤드로 "바쁜 척"만 하고 있는 상태.
       n=100,000 부근부터 처리 속도가 270개/초 -> 2개/초까지 계단식으로 폭락.

v5 수정 사항:
    1. (핵심) 배치를 "개수" 기준이 아니라 "배치 내 (최대 길이 x 개수)" 기준으로
       동적으로 구성 (DynamicDurationBatchSampler).
       - 짧은 오디오: 기존처럼 최대 BATCH_SIZE_CAP(32)개까지 크게 묶음
       - 긴 오디오: 자동으로 배치 크기를 줄여서 VRAM 사용량을 일정하게 유지
       - 이미 duration_ms 순 정렬이 되어 있으므로, 정렬 순서를 그대로 활용해서
         "배치의 마지막(가장 긴) 원소 길이 x 배치 크기"만 확인하면 됨.
    2. torch.no_grad() -> torch.inference_mode() (오토그라드 오버헤드 추가 제거)
    3. 20배치마다 torch.cuda.empty_cache() 호출 (파편화 완화, 매 배치마다 하면
       오버헤드가 누적되므로 주기적으로만 호출)
    4. MAX_BATCH_AUDIO_SECONDS 값은 하드웨어(VRAM 용량)에 따라 다르므로,
       재시작 후 nvidia-smi로 Memory-Usage를 보면서 튜닝 필요:
         - 여전히 VRAM이 꽉 차면 -> 값을 줄이기 (예: 32 -> 16)
         - GPU-Util은 낮은데 메모리 여유가 많으면 -> 값을 늘려서 처리량 개선 가능
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


# =========================================================
# 경로 / 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

WORK_DIR = BASE_DIR / "Work"
COMMON_DIR = WORK_DIR / "Common"

METADATA_PATH = COMMON_DIR / "metadata.csv"
SEGMENT_DIR = COMMON_DIR / "caller_segments"

EMB_DIR = COMMON_DIR / "embeddings"
SHARD_DIR = EMB_DIR / "shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
SHARD_SIZE = 4000
NUM_WORKERS = 4
PREFETCH_FACTOR = 4

# ---- v5: 동적 배치 설정 -----------------------------------
# "배치 내 최대 길이(초) x 배치 크기"가 이 값을 넘지 않도록 배치를 구성.
# 예: 32로 두면 -> 1초짜리 오디오는 32개, 4초짜리는 8개, 16초짜리는 2개씩 묶임.
# RTX 3050 4GB 기준 보수적으로 시작한 값이며, 재시작 후 nvidia-smi로
# VRAM 사용량을 보면서 조정 권장.
MAX_BATCH_AUDIO_SECONDS = 32
BATCH_SIZE_CAP = 32  # 짧은 오디오라도 이 개수 이상으로는 묶지 않음 (기존 동작 유지)

EMPTY_CACHE_EVERY_N_BATCHES = 20  # 파편화 완화용 주기적 캐시 정리

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_SAVEDIR = BASE_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"


# =========================================================
# 오디오 로드 (워커 프로세스에서 실행)
# =========================================================

def load_and_resample(path_str):
    wav, sr = torchaudio.load(path_str)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    return wav.squeeze(0)


class SegmentDataset(Dataset):
    """이미 길이순으로 정렬된 df를 그대로 순서대로 처리"""

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
    batch = sorted(batch, key=lambda x: x[0])  # idx 순서 보장

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


# =========================================================
# duration 기반 동적 배치 샘플러 (v5 신규)
# =========================================================

class DynamicDurationBatchSampler(Sampler):
    """
    데이터셋이 이미 duration 오름차순으로 정렬되어 있다는 전제 하에,
    "배치 내 최대 길이(sample 수) x 배치 크기"가 max_batch_samples를
    넘지 않도록 배치를 동적으로 구성한다.

    정렬되어 있으므로 배치에 마지막으로 추가되는 원소가 항상 그 배치의
    최대 길이가 된다 -> 매 원소 추가 시 (현재 개수+1) x (그 원소 길이)만
    확인하면 됨 (O(n)).
    """

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
                candidate_len = self.durations_samples[j]  # 정렬됨 -> 항상 최대
                candidate_count = len(batch) + 1
                if candidate_count * candidate_len > self.max_batch_samples:
                    break
                batch.append(j)
                j += 1
            yield batch
            i = j

    def __len__(self):
        # 진행률 표시 등에 참고용으로만 쓰이는 대략적인 배치 개수 (정확하지 않아도 무방)
        if not self.durations_samples:
            return 0
        avg_len = sum(self.durations_samples) / len(self.durations_samples)
        est_batch_size = max(1, min(self.batch_size_cap, int(self.max_batch_samples / max(1, avg_len))))
        return (len(self.durations_samples) + est_batch_size - 1) // est_batch_size


# =========================================================
# 메인 로직
# =========================================================

def main():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda:0":
        print("GPU 사용:", torch.cuda.get_device_name(0))
    else:
        print("GPU를 찾지 못해 CPU로 진행합니다.")

    print("ECAPA-TDNN 모델 로드 중...")

    kwargs = dict(
        source=MODEL_SOURCE,
        savedir=str(MODEL_SAVEDIR),
        run_opts={"device": device},
    )
    if _LOCAL_STRATEGY is not None:
        kwargs["local_strategy"] = _LOCAL_STRATEGY

    classifier = EncoderClassifier.from_hparams(**kwargs)
    classifier.eval()

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"metadata.csv가 없습니다: {METADATA_PATH}\n"
            f"02_common_preprocess.py를 먼저 실행하세요."
        )

    df = pd.read_csv(METADATA_PATH)
    total_len = len(df)
    print("전체 segment 수:", total_len)

    # 길이(duration_ms) 순으로 정렬 (v4와 동일, 유지)
    sorted_df = df.sort_values(
        "duration_ms",
        kind="mergesort",  # 안정 정렬 (재실행 시 항상 동일한 순서 보장)
    ).reset_index(drop=True)

    print(
        "길이 범위:",
        f"{sorted_df['duration_ms'].min():.0f}ms ~ "
        f"{sorted_df['duration_ms'].max():.0f}ms",
    )

    n_shards = (total_len + SHARD_SIZE - 1) // SHARD_SIZE
    print("shard 개수:", n_shards, f"(shard당 최대 {SHARD_SIZE}개)")

    # 이미 완료된 shard는 건너뛰고 이어서 진행 (v4와 동일, 유지)
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

    # v5: 고정 batch_size 대신 동적 batch_sampler 사용
    max_batch_samples = int(MAX_BATCH_AUDIO_SECONDS * TARGET_SR)
    batch_sampler = DynamicDurationBatchSampler(
        durations_ms=remaining_df["duration_ms"].tolist(),
        sample_rate=TARGET_SR,
        max_batch_samples=max_batch_samples,
        batch_size_cap=BATCH_SIZE_CAP,
    )
    print(
        f"동적 배치 설정: 배치당 최대 {MAX_BATCH_AUDIO_SECONDS}초 분량(패딩 기준), "
        f"최대 {BATCH_SIZE_CAP}개까지"
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

    with torch.inference_mode():  # v5: no_grad -> inference_mode
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

            # v5: 주기적으로 캐시 정리 (파편화 완화)
            if device == "cuda:0" and batch_counter % EMPTY_CACHE_EVERY_N_BATCHES == 0:
                torch.cuda.empty_cache()

            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(
                f"\r{remaining_start + processed}/{total_len} segment | "
                f"경과 {elapsed / 60:.1f}분 | {rate:.1f}개/초 | "
                f"현재 배치 크기 {len(idxs)}",
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
    print("임베딩 추출 완료")
    print("=" * 60)
    print("총 실패 segment:", failed_count)
    print("저장 위치:", SHARD_DIR)


if __name__ == "__main__":
    main()
