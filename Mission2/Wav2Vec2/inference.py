import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from collections import defaultdict

# subprocess/파이프로 실행돼도 print()가 버퍼에 갇혀서
# 진행 상황이 안 보이는 문제 방지 (줄 단위로 즉시 출력).
# Jupyter/Colab 커널의 sys.stdout(ipykernel OutStream)은
# reconfigure()가 없어서 노트북에서 import할 때만 에러가 나므로 방어.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification
)

from wav2vec2_common import (
    TARGET_SR,
    MIN_DUR_SEC,
    compute_chunks
)


# =========================================================
# Mission 2 실행규칙:
# python inference.py --audio_dir {wav folder} --label_dir {json folder}
#                      --ckpt_path {checkpoint file} --output ./outputs/mission2.csv
#
# 출력 csv: [audio file name], [startAt], [endAt], [speaker]
#           (발화 조각 1개당 1행 - Mission1과 달리 오디오 파일 1개당
#            여러 행이 나옴, 한 통화 안에 발화가 여러 개이므로)
#
# 라벨링데이터는 startAt, endAt만 사용해서 "어디를 잘라 판단할지"를
# 정하고, speaker 필드는 절대 읽지 않는다 (그게 곧 우리가 예측해야
# 할 정답이므로 입력으로 쓰면 정답을 보고 푸는 셈이 됨).
#
# 청크(한 발화 조각이 MAX_DUR_SEC보다 길어 여러 개로 쪼개진 경우)
# 단위 예측 확률을 평균 낸 뒤 argmax로 발화 조각 단위 최종 speaker를
# 결정한다 (학습 때 evaluate()와 동일한 집계 방식 - segment 단위,
# Mission1의 call 단위 집계와는 다름).
# =========================================================

BATCH_SIZE = 16


# =========================================================
# 1. 인자 파싱
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


# =========================================================
# 2. 라벨 json에서 발화 조각(startAt, endAt)만 추출
#    speaker/text 등 다른 필드는 절대 읽지 않음
# =========================================================

def extract_segments(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    utterances = data.get("utterances", [])

    segments = []

    for seg_idx, u in enumerate(utterances):
        raw_start = u.get("startAt")
        raw_end = u.get("endAt")

        if raw_start is None or raw_end is None:
            continue

        try:
            start_ms = float(raw_start)
            end_ms = float(raw_end)
        except (TypeError, ValueError):
            continue

        if end_ms <= start_ms:
            continue

        segments.append({
            "seg_idx": seg_idx,
            "start_sec": start_ms / 1000.0,
            "end_sec": end_ms / 1000.0,
            # 출력 csv에는 json 원본 값을 그대로 씀
            # (float() 변환한 값이 아니라 원래 표기 그대로 보존)
            "raw_start": raw_start,
            "raw_end": raw_end
        })

    return segments


# =========================================================
# 3. audio_dir / label_dir 순회 -> 청크 리스트 생성
#    (01_common_preprocess.py / 02_wav2vec2_prepare.py와 동일한
#     크롭 + 청크 분할 로직을 원본(미분할) wav에 직접 적용)
# =========================================================

def build_chunk_list(audio_dir, label_dir):
    audio_dir = Path(audio_dir)
    label_dir = Path(label_dir)

    wav_files = sorted(audio_dir.glob("*.wav"))

    if not wav_files:
        raise RuntimeError(f"오디오 파일이 없습니다: {audio_dir}")

    call_names = []
    # call_name -> 그 통화의 발화 조각 리스트 (출력 행 순서/개수의 기준)
    segments_by_call = {}
    chunk_list = []

    empty_call_count = 0

    for wav_path in wav_files:
        stem = wav_path.stem
        call_name = wav_path.name
        call_names.append(call_name)

        json_path = label_dir / f"{stem}.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"라벨 파일을 찾을 수 없습니다: {json_path}"
            )

        segments = extract_segments(json_path)
        segments_by_call[call_name] = segments

        if not segments:
            # 발화 조각이 하나도 없는 통화 -> 출력할 행도 없음
            # (Mission1과 달리 call 단위로 1행을 강제할 필요가 없어서
            #  fallback 없이 그냥 건너뜀)
            empty_call_count += 1
            continue

        for seg in segments:
            seg_duration = seg["end_sec"] - seg["start_sec"]
            seg_key = (call_name, seg["seg_idx"])

            for offset_start, offset_end in compute_chunks(seg_duration):
                chunk_list.append({
                    "wav_path": wav_path,
                    "seg_key": seg_key,
                    "chunk_start_sec": seg["start_sec"] + offset_start,
                    "chunk_end_sec": seg["start_sec"] + offset_end
                })

    if empty_call_count:
        print(f"[경고] 발화 조각이 없는 통화: {empty_call_count}개 "
              f"(해당 통화는 출력 행 없음)")

    return call_names, segments_by_call, chunk_list


# =========================================================
# 4. Dataset / collate (학습 때와 동일한 전처리)
# =========================================================

class InferenceChunkDataset(Dataset):

    def __init__(self, chunk_list):
        self.chunk_list = chunk_list

        # 같은 통화에서 나온 청크가 chunk_list에 연속으로 이어져
        # 있으므로(build_chunk_list가 통화 단위로 순회하며 만듦),
        # 방금 읽은 wav 파일 하나만 캐싱해도 같은 통화의 발화 조각을
        # 여러 번 다시 디스크에서 읽는 걸 대부분 피할 수 있다.
        # (num_workers=0으로 고정 - 캐시가 worker마다 따로 생기면
        #  효과가 없으므로, 아래 run_inference 참고)
        self._cache_path = None
        self._cache_waveform = None
        self._cache_sr = None

    def __len__(self):
        return len(self.chunk_list)

    def __getitem__(self, idx):
        item = self.chunk_list[idx]

        if item["wav_path"] == self._cache_path:
            waveform, orig_sr = self._cache_waveform, self._cache_sr
        else:
            # torchaudio.load 대신 soundfile로 직접 읽는다.
            # 최근 torchaudio(2.9+)는 기본 로딩 백엔드가 torchcodec으로
            # 바뀌었는데, 채점 환경에 torchcodec이 없을 수도 있어서
            # (실제로 로컬 테스트 중 ModuleNotFoundError로 확인됨)
            # 이미 전처리 전 과정에서 쓰고 있는 soundfile로 통일해
            # 이 의존성 자체를 없앤다. 리샘플(아래)은 순수 텐서 연산이라
            # 백엔드와 무관하게 항상 동작함.
            y, orig_sr = sf.read(str(item["wav_path"]), always_2d=False)

            if y.ndim > 1:
                y = y.mean(axis=1)

            waveform = torch.from_numpy(y).float().unsqueeze(0)

            self._cache_path = item["wav_path"]
            self._cache_waveform = waveform
            self._cache_sr = orig_sr

        start = int(item["chunk_start_sec"] * orig_sr)
        end = int(item["chunk_end_sec"] * orig_sr)
        end = min(end, waveform.shape[1])
        start = min(start, end)

        clip = waveform[:, start:end]

        if orig_sr != TARGET_SR:
            clip = torchaudio.functional.resample(
                clip, orig_sr, TARGET_SR
            )

        min_samples = int(MIN_DUR_SEC * TARGET_SR)
        if clip.shape[1] < min_samples:
            pad = min_samples - clip.shape[1]
            clip = torch.nn.functional.pad(clip, (0, pad))

        return {
            "input_values": clip.squeeze(0).numpy(),
            "seg_key": item["seg_key"]
        }


def make_collate_fn(feature_extractor):

    def collate_fn(batch):
        input_values = [b["input_values"] for b in batch]

        features = feature_extractor(
            input_values,
            sampling_rate=TARGET_SR,
            padding=True,
            return_tensors="pt"
        )

        seg_keys = [b["seg_key"] for b in batch]

        return {
            "input_values": features["input_values"],
            # base(group-norm) 모델은 attention_mask가 없을 수 있음
            "attention_mask": features.get("attention_mask"),
            "seg_keys": seg_keys
        }

    return collate_fn


# =========================================================
# 5. 추론 -> 발화 조각(segment) 단위 확률 평균 집계
# =========================================================

@torch.no_grad()
def run_inference(model, feature_extractor, chunk_list, device):
    dataset = InferenceChunkDataset(chunk_list)

    # num_workers=0 (기본값): CUDA 컨텍스트가 이미 초기화된 상태에서
    # DataLoader worker를 fork하면 멈추는 알려진 문제가 있어서
    # 안전하게 단일 프로세스로 로딩한다. (Dataset의 1개짜리 wav 캐시가
    # 제대로 동작하려면 어차피 단일 프로세스 + shuffle=False여야 함)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(feature_extractor)
    )

    segment_probs = defaultdict(list)

    total_batches = len(loader)

    for step, batch in enumerate(loader, start=1):
        input_values = batch["input_values"].to(device)

        attention_mask = batch["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask
        )

        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        for seg_key, prob in zip(batch["seg_keys"], probs):
            segment_probs[seg_key].append(prob)

        # 진행 상황 로그
        if step % 50 == 0 or step == total_batches:
            print(f"처리 중: 배치 {step}/{total_batches}")

    return segment_probs


# =========================================================
# 6. 메인
# =========================================================

def main():
    args = parse_args()

    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # -----------------------------------------------------
    # 체크포인트 로드
    # -----------------------------------------------------
    # ckpt_path가 네트워크 드라이브(Google Drive mount 등)를 직접
    # 가리키는 경우 읽기가 느릴 수 있어서, 먼저 로컬 디스크로
    # 복사한 뒤 거기서 읽는다.
    # -----------------------------------------------------

    ckpt_src = Path(args.ckpt_path)
    local_ckpt_path = Path("/tmp") / ckpt_src.name

    if str(ckpt_src) != str(local_ckpt_path):
        print(f"체크포인트 로컬로 복사 중: {ckpt_src} -> {local_ckpt_path}")
        local_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt_src, local_ckpt_path)

    print(f"[{time.time() - t0:.1f}s] 체크포인트 복사 완료, 로드 시작")

    checkpoint = torch.load(local_ckpt_path, map_location=device)

    print(f"[{time.time() - t0:.1f}s] 체크포인트 로드 완료")

    # 전체 pretrained 가중치를 다시 다운로드하지 않고
    # config만 받아서 빈 모델을 만든 뒤, 우리 체크포인트로 채운다
    # (다운로드 시간/네트워크 의존성 줄이기 위함)
    config = AutoConfig.from_pretrained(
        checkpoint["pretrained_ckpt"],
        num_labels=checkpoint["num_labels"]
    )
    model = Wav2Vec2ForSequenceClassification(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[{time.time() - t0:.1f}s] 모델 준비 완료")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        checkpoint["pretrained_ckpt"]
    )

    print(
        "체크포인트 val_segment_level_acc:",
        checkpoint.get("val_segment_level_acc")
    )

    # -----------------------------------------------------
    # 청크 리스트 생성 + 추론
    # -----------------------------------------------------

    call_names, segments_by_call, chunk_list = build_chunk_list(
        args.audio_dir,
        args.label_dir
    )

    total_segments = sum(
        len(segments_by_call[c]) for c in call_names
    )

    print(
        f"[{time.time() - t0:.1f}s] "
        f"오디오 파일 수: {len(call_names)}, "
        f"발화 조각 수: {total_segments}, "
        f"청크 수: {len(chunk_list)}"
    )

    segment_probs = run_inference(
        model,
        feature_extractor,
        chunk_list,
        device
    )

    print(f"[{time.time() - t0:.1f}s] 추론 완료")

    # -----------------------------------------------------
    # segment 단위로 확률 평균 -> argmax -> speaker -> csv
    # -----------------------------------------------------

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        for call_name in call_names:
            for seg in segments_by_call[call_name]:
                seg_key = (call_name, seg["seg_idx"])
                probs = segment_probs.get(seg_key)

                if not probs:
                    raise RuntimeError(
                        f"청크를 하나도 못 만든 발화 조각: {seg_key}"
                    )

                mean_prob = np.mean(probs, axis=0)
                pred_speaker = int(np.argmax(mean_prob))

                writer.writerow([
                    call_name,
                    seg["raw_start"],
                    seg["raw_end"],
                    pred_speaker
                ])

                n_rows += 1

    print(f"출력 행 수: {n_rows}")
    print("완료:", output_path)


if __name__ == "__main__":
    main()
