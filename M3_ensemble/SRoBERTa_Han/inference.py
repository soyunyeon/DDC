# ============================================================
# Mission 3 - inference.py (SRoBERTa, 발화 단위 pooling + partial fine-tuning)
#
# 실행 (대회 지정 형식):
#   python inference.py --audio_dir {wav folder} --label_dir {json folder} \
#       --ckpt_path {checkpoint file} --output ./outputs/mission3.csv
#
# - Mission 3는 음성을 쓰지 않으므로 --audio_dir은 받기만 하고 사용하지 않는다.
# - 라벨링 JSON에서는 utterances[].text만 읽는다.
#   symptom, speaker, startAt, endAt, gender 등 다른 항목은 읽지도 사용하지도 않는다.
# - Decision Threshold는 대회 규정에 따라 0.5로 고정한다.
# - 인터넷 없이 실행된다. tokenizer와 config는 assets/ 폴더에서 읽는다.
#
# 출력 CSV: label_file_name, symptom
#   symptom은 예측된 증상 리스트를 문자열로 저장한다. 예: ['두통', '복통']
#   예측된 증상이 없으면 [] 로 저장한다.
#
# 선택 옵션
#   --prob_output : 증상별 sigmoid 확률(threshold 적용 전)을 CSV로 같이 저장 (앙상블용)
#   --batch_size  : 추론 batch size (기본 16)
# ============================================================

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from model import HierarchicalClassifier


HERE = Path(__file__).resolve().parent


# ============================================================
# 1. 설정 (학습 때와 동일해야 함)
# ============================================================

# 모델 출력 9차원의 순서. 학습 때 순서와 반드시 같아야 한다.
TARGET_SYMPTOMS = [
    "고열", "구토", "두통", "복통", "어지러움",
    "열상", "오심", "전신쇠약", "호흡곤란",
]

MAX_UTT = 16                # 콜당 최대 발화 수
MAX_LEN = 64                # 발화 하나당 최대 token 길이
DECISION_THRESHOLD = 0.5    # 대회 규정 고정값, 변경 금지


# ============================================================
# 2. 전처리 (학습 때 공통 전처리와 동일)
# ============================================================

def clean_text(text):
    """앞뒤 공백 제거, 연속 공백/탭을 공백 하나로 정리. 의미를 바꾸는 처리는 하지 않는다."""

    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)

    return text


def load_utterances(json_path):
    """JSON 하나에서 utterances[].text만 뽑아서 빈 발화를 제외한 리스트로 반환."""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []

    for utt in data.get("utterances", []):
        text = clean_text(utt.get("text", ""))
        if text:
            texts.append(text)

    return texts


# ============================================================
# 3. Dataset
# ============================================================

class CallDataset(Dataset):

    def __init__(self, utterances_list, tokenizer, max_utt, max_len):

        self.utterances_list = utterances_list
        self.tokenizer = tokenizer
        self.max_utt = max_utt
        self.max_len = max_len

    def __len__(self):
        return len(self.utterances_list)

    def __getitem__(self, idx):

        utts = list(self.utterances_list[idx])

        if len(utts) == 0:
            utts = [""]

        # 발화가 너무 많으면 긴 발화 우선으로 max_utt개만 사용 (학습 때와 동일)
        if len(utts) > self.max_utt:
            utts = sorted(utts, key=len, reverse=True)[: self.max_utt]

        n_real = len(utts)
        utts_padded = utts + [""] * (self.max_utt - n_real)

        enc = self.tokenizer(
            utts_padded,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        utt_mask = torch.zeros(self.max_utt)
        utt_mask[:n_real] = 1.0

        return enc["input_ids"], enc["attention_mask"], utt_mask


# ============================================================
# 4. 실행
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(description="Mission 3 inference (SRoBERTa)")

    parser.add_argument("--audio_dir", type=str, default=None,
                        help="사용하지 않음 (대회 실행 규격 호환용)")
    parser.add_argument("--label_dir", type=str, required=True,
                        help="라벨링 JSON이 들어 있는 폴더")
    parser.add_argument("--ckpt_path", type=str,
                        default=str(HERE / "checkpoints" / "best_model.pt"),
                        help="학습된 체크포인트 파일 (.pt)")
    parser.add_argument("--output", type=str, default="./outputs/mission3.csv",
                        help="결과 CSV 경로")
    parser.add_argument("--assets_dir", type=str,
                        default=str(HERE / "assets"),
                        help="tokenizer, config 파일이 있는 폴더")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--prob_output", type=str, default=None,
                        help="(선택) 증상별 확률 CSV 저장 경로")

    return parser.parse_args()


def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    print(f"Device: {device} ({device_name})")

    # --------------------------------------------------------
    # 입력 JSON 수집 (파일명 순으로 정렬해서 출력 순서를 고정)
    # --------------------------------------------------------

    label_dir = Path(args.label_dir)
    json_files = sorted(label_dir.rglob("*.json"))

    if len(json_files) == 0:
        raise FileNotFoundError(f"JSON 파일이 없습니다: {label_dir}")

    print(f"JSON 파일 수: {len(json_files)}")

    file_names = []
    utterances_list = []
    n_failed = 0

    for p in json_files:

        file_names.append(p.name)

        try:
            utterances_list.append(load_utterances(p))
        except Exception as e:
            # 읽기에 실패해도 행 수가 어긋나지 않도록 빈 발화로 처리
            print(f"[WARNING] JSON 읽기 실패: {p.name} ({e})")
            utterances_list.append([])
            n_failed += 1

    if n_failed:
        print(f"[WARNING] 읽기 실패 {n_failed}건은 빈 대화로 처리했습니다.")

    # --------------------------------------------------------
    # tokenizer / 모델 로드 (전부 로컬 파일, 인터넷 불필요)
    # --------------------------------------------------------

    assets_dir = Path(args.assets_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(assets_dir), local_files_only=True)

    model = HierarchicalClassifier(assets_dir, num_labels=len(TARGET_SYMPTOMS))

    state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total_params:,}")
    print(f"Batch size: {args.batch_size}")

    dataset = CallDataset(utterances_list, tokenizer, MAX_UTT, MAX_LEN)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --------------------------------------------------------
    # 추론
    # --------------------------------------------------------

    all_probs = []

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():

        for step, (input_ids, attention_mask, utt_mask) in enumerate(loader, start=1):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            utt_mask = utt_mask.to(device)

            logits = model(input_ids, attention_mask, utt_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)

            if step % 50 == 0:
                print(f"  {step}/{len(loader)} batch 완료")

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    probs = np.concatenate(all_probs, axis=0)

    print(f"전체 추론시간: {elapsed:.2f}초")
    print(f"샘플당 평균 추론시간: {elapsed / len(file_names) * 1000:.2f}ms")

    # --------------------------------------------------------
    # threshold 0.5 고정으로 라벨 변환 후 저장
    # --------------------------------------------------------

    preds = probs >= DECISION_THRESHOLD

    symptom_strings = []

    for row in preds:
        predicted = [s for s, flag in zip(TARGET_SYMPTOMS, row) if flag]
        symptom_strings.append(str(predicted))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame({
        "label_file_name": file_names,
        "symptom": symptom_strings,
    })

    result_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"결과 저장: {out_path}")

    if args.prob_output:

        prob_path = Path(args.prob_output)
        prob_path.parent.mkdir(parents=True, exist_ok=True)

        prob_df = pd.DataFrame(probs, columns=[f"{s}_prob" for s in TARGET_SYMPTOMS])
        prob_df.insert(0, "label_file_name", file_names)
        prob_df.to_csv(prob_path, index=False, encoding="utf-8")

        print(f"확률값 저장: {prob_path}")

    # 간단한 sanity check
    n_pred = preds.sum(axis=1)
    print(f"평균 예측 증상 수: {n_pred.mean():.3f} / 0개 예측: {(n_pred == 0).sum()}건")


if __name__ == "__main__":
    main()
