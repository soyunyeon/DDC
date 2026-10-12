# =========================================================
# Mission 2 - Fusion(4번) Cross-Attention 실험용
# frame/sequence-level(pooling 전) audio feature 추출
#
# 07_export_fusion_predictions.ipynb가 만든 pooled(256-d, 시간축
# 평균) 임베딩과 달리, 분류기 pooling 바로 직전의 "프레임 단위"
# 시퀀스 특징을 발화 조각(segment)별 .npy로 저장한다. Text 토큰과의
# Cross-Attention 실험용 - Fusion(4번) 담당자가 자신의 서버에서
# 직접 실행하는 용도로 작성함 (Colab/Drive 의존성 없이 독립 실행
# 가능하도록 07_export_fusion_predictions.ipynb 로직을 재구성함).
#
# 체크포인트: mission2_wav2vec2_best.pt
#   (N_UNFROZEN_LAYERS=4, 컨텍스트 마진 없음 - 최신/최종 제출 버전.
#    실험2(±0.3초 마진)는 대회 규정 위반으로 폐기됨 - model.py 참고)
# 사전학습 모델: Kkonjeong/wav2vec2-base-korean
#   (체크포인트 안에 pretrained_ckpt로 기록되어 있어 자동으로 맞춰짐)
#
# 입력: Mission2/Work/Common/metadata.csv + segments/*.wav
#   (01_common_preprocess.py 산출물 - call_id, segment_id,
#    segment_file, speaker, split 컬럼 사용)
# 출력: --output_dir 아래
#   - frames/{call_id}_{segment_id}_{chunk_index}.npy
#     (shape [T, D], float16 - T=프레임 수, D=256(projector) 또는 768(encoder))
#   - manifest.csv (call_id, segment_id, chunk_index, speaker, split,
#     n_frames, feature_dim, npy_path)
#
# 주의 1) 청크 분할: wav2vec2_common.MAX_DUR_SEC(8초)보다 긴 발화
#   조각은 학습/추론 때와 동일하게 50% overlap sliding window로 여러
#   청크로 쪼개진다 (compute_chunks). 대부분(8초 이하)은 청크가
#   1개뿐이라 문제 없지만, 8초 넘는 소수 발화 조각은 chunk_index가
#   0,1,2...로 늘어나며 프레임 시퀀스가 여러 개로 나뉜다. 이걸
#   cross-attention에서 어떻게 합칠지(첫 청크만 사용/겹치는 4초
#   구간을 잘라내고 concat 등)는 Fusion 쪽에서 결정 필요 - 이 스크립트는
#   병합하지 않고 그대로 각각 저장만 한다.
#
# 주의 2) 파일 개수: 전체(889,358 세그먼트)를 그대로 돌리면 .npy
#   파일이 90만개 가까이 생긴다. 작은 파일이 매우 많아지므로 파일시스템
#   성능이 걱정되면 --limit으로 일부만 먼저 테스트하거나, 여러 세그먼트를
#   하나의 파일(npz/HDF5 등)로 묶는 방식으로 이 스크립트를 고쳐서 쓸 것.
#
# 주의 3) 배치 크기 1로 세그먼트/청크를 하나씩 처리한다 (프레임 단위
#   출력은 세그먼트마다 길이가 달라 배치로 묶으려면 패딩 + 프레임 단위
#   attention_mask 처리가 필요함 - 정확성 우선으로 일단 단순하게 작성함,
#   속도가 문제면 길이별로 묶어서 배치 처리하도록 확장 가능).
# =========================================================

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification
)

from wav2vec2_common import TARGET_SR, MIN_DUR_SEC, compute_chunks


FEATURE_LEVELS = ("projector", "encoder")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", default="Mission2/Work/Common/metadata.csv")
    parser.add_argument("--segments_dir", default="Mission2/Work/Common/segments")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument(
        "--feature_level", choices=FEATURE_LEVELS, default="projector",
        help=(
            "projector = classifier 직전 256-d 프레임 특징 "
            "(projector를 통과했지만 시간축 pooling은 하지 않은 것 - "
            "기존에 전달한 256-d pooled 임베딩과 차원은 같고 pooling만 뺀 버전). "
            "encoder = wav2vec2 인코더의 raw 768-d 프레임 특징 "
            "(projector를 통과하기 전 hidden state)"
        )
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="테스트용: metadata.csv 앞에서 N개 세그먼트만 처리"
    )
    return parser.parse_args()


def load_segment_waveform(wav_path):
    y, orig_sr = sf.read(str(wav_path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    waveform = torch.from_numpy(y).float().unsqueeze(0)
    return waveform, orig_sr


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)

    config = AutoConfig.from_pretrained(
        checkpoint["pretrained_ckpt"], num_labels=checkpoint["num_labels"]
    )
    assert not config.use_weighted_layer_sum, (
        "use_weighted_layer_sum=True인 체크포인트 - 이 스크립트의 "
        "pooling-생략 로직을 이 경우에 맞게 다시 확인해야 함"
    )

    model = Wav2Vec2ForSequenceClassification(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        checkpoint["pretrained_ckpt"]
    )

    print("pretrained_ckpt:", checkpoint["pretrained_ckpt"])
    print("체크포인트 val_segment_level_acc:", checkpoint.get("val_segment_level_acc"))

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    segments_dir = Path(args.segments_dir)

    with open(args.metadata_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]

    if args.limit:
        rows = rows[: args.limit]

    print(f"대상 세그먼트: {len(rows)}개 (feature_level={args.feature_level})")

    manifest_rows = []

    with torch.no_grad():
        for i, row in enumerate(rows, start=1):
            wav_path = segments_dir / row["segment_file"]
            waveform, orig_sr = load_segment_waveform(wav_path)

            duration_sec = waveform.shape[1] / orig_sr

            for chunk_index, (offset_start, offset_end) in enumerate(compute_chunks(duration_sec)):
                start = int(offset_start * orig_sr)
                end = int(offset_end * orig_sr)
                clip = waveform[:, start:end]

                if orig_sr != TARGET_SR:
                    clip = torchaudio.functional.resample(clip, orig_sr, TARGET_SR)

                min_samples = int(MIN_DUR_SEC * TARGET_SR)
                if clip.shape[1] < min_samples:
                    clip = torch.nn.functional.pad(clip, (0, min_samples - clip.shape[1]))

                features = feature_extractor(
                    [clip.squeeze(0).numpy()], sampling_rate=TARGET_SR,
                    padding=True, return_tensors="pt"
                )
                input_values = features["input_values"].to(device)
                attention_mask = features.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                encoder_out = model.wav2vec2(
                    input_values=input_values, attention_mask=attention_mask
                )

                if args.feature_level == "encoder":
                    frame_feats = encoder_out.last_hidden_state  # [1, T, 768]
                else:
                    frame_feats = model.projector(encoder_out.last_hidden_state)  # [1, T, 256]

                frame_feats = frame_feats.squeeze(0).cpu().numpy().astype(np.float16)  # [T, D]

                out_name = f"{row['call_id']}_{row['segment_id']}_{chunk_index}.npy"
                np.save(frames_dir / out_name, frame_feats)

                manifest_rows.append([
                    row["call_id"], row["segment_id"], chunk_index,
                    row["speaker"], row["split"],
                    frame_feats.shape[0], frame_feats.shape[1],
                    f"frames/{out_name}"
                ])

            if i % 500 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} 세그먼트 처리 완료")

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "call_id", "segment_id", "chunk_index", "speaker", "split",
            "n_frames", "feature_dim", "npy_path"
        ])
        writer.writerows(manifest_rows)

    print("manifest 저장 완료:", manifest_path)


if __name__ == "__main__":
    main()
