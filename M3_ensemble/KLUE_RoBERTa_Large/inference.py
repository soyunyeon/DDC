# ============================================================
# Mission 3 - klue_roberta_large (2번 후보) 단독 inference
#
# Best model: klue/roberta-large, multi-label fine-tuning
# (AutoModelForSequenceClassification, problem_type=
# "multi_label_classification"), 오심에만 선택적 pos_weight 적용,
# threshold 0.5 고정.
#
# IMPORTANT
# - utterances[].text만 사용 (symptom 등 다른 필드는 읽지 않음)
# - threshold는 클래스별로 다르게 쓰지 않고 0.5 고정
#
# 주의: --ckpt_path는 단일 .pt 파일이 아니라 best_model 폴더
# 경로입니다 (config.json / model.safetensors / tokenizer.json /
# tokenizer_config.json이 한 폴더 안에 같이 들어있는 HuggingFace
# save_pretrained 형식). 예:
#   --ckpt_path .../klue_roberta_large_posweight_selective/best_model
#
# 사용 예:
#   python inference.py \
#       --label_dir /path/to/label_json_dir \
#       --ckpt_path /path/to/best_model \
#       --output ./outputs/mission3.csv
# ============================================================

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


TARGET_SYMPTOMS = [
    "고열",
    "구토",
    "두통",
    "복통",
    "어지러움",
    "열상",
    "오심",
    "전신쇠약",
    "호흡곤란",
]

MAX_LENGTH = 512
THRESHOLD = 0.5
BATCH_SIZE = 32


# ============================================================
# 텍스트 전처리 (01_common_preprocess.py의 clean_text와 동일 -
# 학습 때 쓴 전처리와 추론 때 전처리가 달라지면 안 됨)
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)

    return text


def parse_json(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []

    for utt in data.get("utterances", []):

        text = clean_text(utt.get("text", ""))

        if text:
            texts.append(text)

    full_text = "\n".join(texts)

    return {
        "label_file_name": json_path.name,
        "text": full_text,
    }


def find_json_files(label_dir):

    label_dir = Path(label_dir)

    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir not found: {label_dir}")

    files = sorted(label_dir.rglob("*.json"))

    if not files:
        raise RuntimeError(f"No JSON files found in {label_dir}")

    return files


def main():

    parser = argparse.ArgumentParser()

    # 공식 실행 인터페이스 유지 - Mission 3는 audio_dir을 쓰지 않지만
    # 주최 측 명령어 형식을 맞추기 위해 인자는 받는다.
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True, help="best_model 폴더 경로")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)

    args = parser.parse_args()

    start_time = time.time()

    ckpt_path = Path(args.ckpt_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Mission 3 Inference - klue_roberta_large")
    print("=" * 70)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", ckpt_path)
    print("Label directory:", args.label_dir)
    print("Output:", output_path)
    print("MAX_LENGTH:", args.max_length)
    print("Threshold:", THRESHOLD)

    # ========================================================
    # 모델 / 토크나이저 로드 (config.json, model.safetensors,
    # tokenizer.json, tokenizer_config.json이 전부 ckpt_path 폴더
    # 안에 있어야 함)
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    model.to(device)
    model.eval()

    # ========================================================
    # JSON 파일 로드 및 전처리
    # ========================================================

    json_files = find_json_files(args.label_dir)
    print("\nJSON files:", len(json_files))

    samples = [parse_json(p) for p in tqdm(json_files, desc="Preprocessing")]

    # ========================================================
    # 추론 (배치 단위)
    # ========================================================

    result_rows = []

    with torch.inference_mode():

        for start in tqdm(range(0, len(samples), args.batch_size), desc="Inference"):

            batch = samples[start:start + args.batch_size]
            batch_texts = [s["text"] for s in batch]
            batch_filenames = [s["label_file_name"] for s in batch]

            encoding = tokenizer(
                batch_texts,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)

            logits = model(**encoding).logits
            probabilities = torch.sigmoid(logits)

            # 대회 규정: probability >= 0.5 -> 1, 클래스별 threshold 없음
            predictions = (probabilities >= THRESHOLD).cpu().numpy()

            for filename, pred_vector in zip(batch_filenames, predictions):

                predicted_symptoms = [
                    TARGET_SYMPTOMS[i]
                    for i, value in enumerate(pred_vector)
                    if bool(value)
                ]

                result_rows.append({
                    "label file name": filename,
                    "symptom": str(predicted_symptoms),
                })

    # ========================================================
    # CSV 저장
    # ========================================================

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(f, fieldnames=["label file name", "symptom"])
        writer.writeheader()
        writer.writerows(result_rows)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("Inference completed")
    print("=" * 70)
    print("Samples:", len(result_rows))
    print("Output:", output_path)
    print(f"Total inference time: {elapsed:.2f} sec")
    if len(result_rows) > 0:
        print(f"Average per sample: {elapsed / len(result_rows):.6f} sec")


if __name__ == "__main__":
    main()
