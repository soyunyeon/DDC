import argparse
import ast
import csv
import json
import re
import time
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
)

from tqdm import tqdm


# ============================================================
# Mission 3
# Patient Symptom Multi-label Classification
#
# Best model:
# KLUE-RoBERTa-base
# 512-token truncation
# CLS classification
#
# IMPORTANT
# - Only utterances[].text is used.
# - No speaker / startAt / endAt / gender / triage etc.
# - Decision threshold is fixed at 0.5.
# ============================================================


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

BATCH_SIZE = 8


# ============================================================
# Text preprocessing
#
# Training preprocessing과 동일하게
# 의미를 변경하는 전처리는 하지 않는다.
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.strip()

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text


# ============================================================
# JSON → text
#
# Mission 3 규정:
# utterances[].text만 사용
#
# symptom을 포함한 다른 annotation은 읽지 않는다.
# ============================================================

def parse_json(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    texts = []

    for utt in data.get(
        "utterances",
        [],
    ):

        text = clean_text(
            utt.get(
                "text",
                "",
            )
        )

        if text:
            texts.append(text)

    full_text = "\n".join(
        texts
    )

    return {
        "label_file_name":
            json_path.name,

        "text":
            full_text,
    }


# ============================================================
# Label JSON 탐색
# ============================================================

def find_json_files(label_dir):

    label_dir = Path(
        label_dir
    )

    if not label_dir.exists():

        raise FileNotFoundError(
            f"label_dir not found: "
            f"{label_dir}"
        )

    files = sorted(
        label_dir.rglob(
            "*.json"
        )
    )

    if not files:

        raise RuntimeError(
            "No JSON files found in "
            f"{label_dir}"
        )

    return files


# ============================================================
# Dataset
# ============================================================

class Mission3InferenceDataset(
    Dataset
):

    def __init__(
        self,
        json_files,
        tokenizer,
    ):

        self.samples = []

        print(
            f"Preparing "
            f"{len(json_files)} JSON files..."
        )

        for json_path in tqdm(
            json_files,
            desc="Preprocessing",
        ):

            sample = parse_json(
                json_path
            )

            encoded = tokenizer(
                sample["text"],

                max_length=MAX_LENGTH,

                truncation=True,

                add_special_tokens=True,

                return_attention_mask=True,
            )

            self.samples.append(
                {
                    "input_ids":
                        encoded[
                            "input_ids"
                        ],

                    "attention_mask":
                        encoded[
                            "attention_mask"
                        ],

                    "label_file_name":
                        sample[
                            "label_file_name"
                        ],
                }
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx,
    ):

        return self.samples[
            idx
        ]


# ============================================================
# Collator
# ============================================================

class InferenceCollator:

    def __init__(
        self,
        tokenizer,
    ):

        self.tokenizer = (
            tokenizer
        )

    def __call__(
        self,
        batch,
    ):

        features = []

        filenames = []

        for sample in batch:

            features.append(
                {
                    "input_ids":
                        sample[
                            "input_ids"
                        ],

                    "attention_mask":
                        sample[
                            "attention_mask"
                        ],
                }
            )

            filenames.append(
                sample[
                    "label_file_name"
                ]
            )

        padded = (
            self.tokenizer.pad(
                features,

                padding=True,

                return_tensors="pt",
            )
        )

        return {
            "input_ids":
                padded[
                    "input_ids"
                ],

            "attention_mask":
                padded[
                    "attention_mask"
                ],

            "label_file_name":
                filenames,
        }


# ============================================================
# Model
#
# 08_train_truncation_baseline.py와 동일
# ============================================================

class TruncationBaselineModel(
    nn.Module
):

    def __init__(
        self,
        model_dir,
    ):

        super().__init__()

        # ----------------------------------------------------
        # best_model.pt 안에 encoder weight 전체가 있으므로
        # pretrained weight를 별도로 로딩할 필요 없음.
        #
        # config로 빈 RoBERTa 구조를 만든 뒤
        # checkpoint 전체 state_dict를 로딩한다.
        # ----------------------------------------------------

        config = (
            AutoConfig.from_pretrained(
                model_dir,
                local_files_only=True,
            )
        )

        self.encoder = (
            AutoModel.from_config(
                config,
                add_pooling_layer=False,
            )
        )

        hidden_size = (
            config.hidden_size
        )

        self.dropout = (
            nn.Dropout(
                0.1
            )
        )

        self.classifier = (
            nn.Linear(
                hidden_size,
                len(
                    TARGET_SYMPTOMS
                ),
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = (
            self.encoder(
                input_ids=input_ids,

                attention_mask=(
                    attention_mask
                ),
            )
        )

        cls_embedding = (
            outputs
            .last_hidden_state[
                :, 0, :
            ]
        )

        cls_embedding = (
            self.dropout(
                cls_embedding
            )
        )

        logits = (
            self.classifier(
                cls_embedding
            )
        )

        return logits


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    # --------------------------------------------------------
    # 공식 실행 인터페이스 유지
    #
    # Mission 3에서는 audio_dir을 사용하지 않지만
    # 주최 측 명령어 형식을 맞추기 위해 argument는 받는다.
    # --------------------------------------------------------

    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
    )

    args = (
        parser.parse_args()
    )

    start_time = (
        time.time()
    )


    # ========================================================
    # Path
    # ========================================================

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    if args.model_dir is None:

        model_dir = (
            script_dir
            / "pretrained"
            / "klue-roberta-base"
        )

    else:

        model_dir = Path(
            args.model_dir
        )

    ckpt_path = Path(
        args.ckpt_path
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 70
    )

    print(
        "Mission 3 Inference"
    )

    print(
        "=" * 70
    )

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "Model assets:",
        model_dir,
    )

    print(
        "Checkpoint:",
        ckpt_path,
    )

    print(
        "Label directory:",
        args.label_dir,
    )

    print(
        "Output:",
        output_path,
    )

    print(
        "MAX_LENGTH:",
        MAX_LENGTH,
    )

    print(
        "Threshold:",
        THRESHOLD,
    )


    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_dir,

            local_files_only=True,
        )
    )


    # ========================================================
    # JSON files
    # ========================================================

    json_files = (
        find_json_files(
            args.label_dir
        )
    )

    print(
        "\nJSON files:",
        len(json_files),
    )


    # ========================================================
    # Dataset / Loader
    # ========================================================

    dataset = (
        Mission3InferenceDataset(
            json_files,
            tokenizer,
        )
    )

    loader = (
        DataLoader(
            dataset,

            batch_size=(
                args.batch_size
            ),

            shuffle=False,

            collate_fn=(
                InferenceCollator(
                    tokenizer
                )
            ),

            num_workers=0,

            pin_memory=(
                torch.cuda.is_available()
            ),
        )
    )


    # ========================================================
    # Model
    # ========================================================

    model = (
        TruncationBaselineModel(
            model_dir
        )
    )


    # ========================================================
    # Checkpoint
    # ========================================================

    checkpoint = (
        torch.load(
            ckpt_path,

            map_location="cpu",

            weights_only=False,
        )
    )


    # --------------------------------------------------------
    # checkpoint 정보 확인
    # --------------------------------------------------------

    ckpt_threshold = (
        checkpoint.get(
            "threshold",
            THRESHOLD,
        )
    )

    if float(
        ckpt_threshold
    ) != 0.5:

        raise RuntimeError(
            "Checkpoint threshold is "
            f"{ckpt_threshold}. "
            "Competition rule requires "
            "threshold=0.5."
        )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],

        strict=True,
    )


    model.to(
        device
    )

    model.eval()


    # ========================================================
    # Inference
    # ========================================================

    result_rows = []


    with torch.inference_mode():

        for batch in tqdm(
            loader,

            desc="Inference",
        ):

            input_ids = (
                batch[
                    "input_ids"
                ]
                .to(
                    device,

                    non_blocking=True,
                )
            )

            attention_mask = (
                batch[
                    "attention_mask"
                ]
                .to(
                    device,

                    non_blocking=True,
                )
            )


            if device.type == "cuda":

                with torch.amp.autocast(
                    device_type="cuda",

                    dtype=torch.float16,
                ):

                    logits = (
                        model(
                            input_ids=(
                                input_ids
                            ),

                            attention_mask=(
                                attention_mask
                            ),
                        )
                    )

            else:

                logits = (
                    model(
                        input_ids=(
                            input_ids
                        ),

                        attention_mask=(
                            attention_mask
                        ),
                    )
                )


            probabilities = (
                torch.sigmoid(
                    logits
                )
            )


            # =================================================
            # 대회 규정:
            #
            # probability >= 0.5 → 1
            # probability <  0.5 → 0
            #
            # class별 threshold 사용하지 않음.
            # =================================================

            predictions = (
                probabilities
                >= THRESHOLD
            )


            predictions = (
                predictions
                .cpu()
                .numpy()
            )


            for (
                filename,
                pred_vector,
            ) in zip(
                batch[
                    "label_file_name"
                ],
                predictions,
            ):

                predicted_symptoms = [

                    TARGET_SYMPTOMS[i]

                    for i, value
                    in enumerate(
                        pred_vector
                    )

                    if bool(value)
                ]


                # ------------------------------------------
                # Mission 3 지정 형식:
                #
                # symptom은 string type
                #
                # 예:
                # ['두통', '복통']
                # ------------------------------------------

                symptom_string = str(
                    predicted_symptoms
                )


                result_rows.append(
                    {
                        "label file name":
                            filename,

                        "symptom":
                            symptom_string,
                    }
                )


    # ========================================================
    # CSV 저장
    # ========================================================

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,

            fieldnames=[
                "label file name",
                "symptom",
            ],
        )

        writer.writeheader()

        writer.writerows(
            result_rows
        )


    # ========================================================
    # 완료
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "Inference completed"
    )

    print(
        "=" * 70
    )

    print(
        "Samples:",
        len(result_rows),
    )

    print(
        "Output:",
        output_path,
    )

    print(
        f"Total inference time: "
        f"{elapsed:.2f} sec"
    )

    if len(result_rows) > 0:

        print(
            f"Average per sample: "
            f"{elapsed / len(result_rows):.6f} sec"
        )


if __name__ == "__main__":

    main()

