import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from tqdm import tqdm


# ============================================================
# 1. 기본 설정
# ============================================================

BASE_DIR = (
    Path.home()
    / "DCC"
    / "Mission3"
)

DATA_DIR = (
    Path.home()
    / "DCC"
    / "M3_JSON"
    / "Work"
    / "M3_Common"
)

TRAIN_PATH = (
    DATA_DIR
    / "mission3_train.csv"
)

DEV_PATH = (
    DATA_DIR
    / "mission3_dev.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "pretrained"
    / "klue-roberta-base"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "truncation_baseline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Mission 3 Target
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

LABEL_COLUMNS = [
    f"label_{x}"
    for x in TARGET_SYMPTOMS
]


# ============================================================
# 3. 학습 설정
# ============================================================

MAX_LENGTH = 512

BATCH_SIZE = 4

GRAD_ACCUM_STEPS = 4

EPOCHS = 3

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.1

DROPOUT = 0.1

# ============================================================
# 대회 규정:
# Decision Threshold는 항상 0.5 고정
# ============================================================

THRESHOLD = 0.5

SEED = 42


# ============================================================
# 4. Seed
# ============================================================

def set_seed(seed):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 5. Dataset
#
# 규정:
# Mission 3 입력은 전사 text만 사용
#
# 이번 baseline에서는
# 512 token을 초과하는 부분을 단순 truncation
# ============================================================

class Mission3TruncationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        desc="Tokenizing",
    ):

        self.samples = []

        print(
            f"\n{desc}: "
            f"{len(dataframe)} samples"
        )

        for _, row in tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc=desc,
        ):

            # ----------------------------------------------
            # 허용 입력:
            # 전사 text만 사용
            # ----------------------------------------------

            text = str(
                row["text"]
            )

            # ----------------------------------------------
            # NO CHUNKING
            #
            # 512 token 초과 시
            # 뒤쪽은 그대로 잘라냄
            # ----------------------------------------------

            encoded = tokenizer(
                text,

                max_length=MAX_LENGTH,

                truncation=True,

                add_special_tokens=True,

                return_attention_mask=True,
            )

            labels = [
                float(
                    row[col]
                )
                for col
                in LABEL_COLUMNS
            ]

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

                    "labels":
                        labels,

                    "sample_key":
                        row[
                            "sample_key"
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
# 6. Collator
# ============================================================

class TruncationCollator:

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

        labels = []

        sample_keys = []

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

            labels.append(
                sample[
                    "labels"
                ]
            )

            sample_keys.append(
                sample[
                    "sample_key"
                ]
            )

        # ----------------------------------------------
        # batch 내 최대 길이에 맞춰 dynamic padding
        # ----------------------------------------------

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

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.float32,
                ),

            "sample_keys":
                sample_keys,
        }


# ============================================================
# 7. Truncation Baseline Model
# ============================================================

class TruncationBaselineModel(
    nn.Module
):

    def __init__(
        self,
        model_path,
    ):

        super().__init__()

        # ----------------------------------------------
        # 인터넷 없이 local pretrained model 사용
        #
        # pooler 사용하지 않고
        # CLS hidden state 직접 사용
        # ----------------------------------------------

        self.encoder = (
            AutoModel.from_pretrained(
                model_path,

                local_files_only=True,

                add_pooling_layer=False,
            )
        )

        hidden_size = (
            self.encoder
            .config
            .hidden_size
        )

        self.dropout = (
            nn.Dropout(
                DROPOUT
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

        # ----------------------------------------------
        # [CLS] representation
        # ----------------------------------------------

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

        # ----------------------------------------------
        # 9 symptom logits
        # ----------------------------------------------

        logits = (
            self.classifier(
                cls_embedding
            )
        )

        return logits


# ============================================================
# 8. Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    loss_fn,
):

    model.eval()

    total_loss = 0.0

    all_probs = []

    all_labels = []

    for batch in tqdm(
        dataloader,

        desc="Validation",

        leave=False,
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

        labels = (
            batch[
                "labels"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        with torch.amp.autocast(
            device_type="cuda",

            dtype=torch.float16,
        ):

            logits = model(
                input_ids=(
                    input_ids
                ),

                attention_mask=(
                    attention_mask
                ),
            )

            loss = (
                loss_fn(
                    logits,
                    labels,
                )
            )

        probs = (
            torch.sigmoid(
                logits
            )
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        all_probs.append(
            probs
            .cpu()
            .numpy()
        )

        all_labels.append(
            labels
            .cpu()
            .numpy()
        )

    probs = (
        np.concatenate(
            all_probs,
            axis=0,
        )
    )

    labels = (
        np.concatenate(
            all_labels,
            axis=0,
        )
        .astype(int)
    )

    # ========================================================
    # 대회 규정
    #
    # 모든 class 동일하게
    # Decision Threshold = 0.5
    # ========================================================

    preds = (
        probs
        >= THRESHOLD
    ).astype(int)

    macro_f1 = (
        f1_score(
            labels,
            preds,

            average="macro",

            zero_division=0,
        )
    )

    per_class_f1 = (
        f1_score(
            labels,
            preds,

            average=None,

            zero_division=0,
        )
    )

    per_class_precision = (
        precision_score(
            labels,
            preds,

            average=None,

            zero_division=0,
        )
    )

    per_class_recall = (
        recall_score(
            labels,
            preds,

            average=None,

            zero_division=0,
        )
    )

    avg_loss = (
        total_loss
        / len(
            dataloader.dataset
        )
    )

    return {

        "loss":
            avg_loss,

        "macro_f1":
            macro_f1,

        "f1":
            per_class_f1,

        "precision":
            per_class_precision,

        "recall":
            per_class_recall,
    }


# ============================================================
# 9. Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )


    set_seed(
        SEED
    )


    device = (
        torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )


    print(
        "=" * 70
    )

    print(
        "Mission 3 "
        "512-token Truncation Baseline"
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
            torch.cuda.get_device_name(0),
        )

    print(
        "Threshold:",
        THRESHOLD,
    )

    print(
        "MAX_LENGTH:",
        MAX_LENGTH,
    )

    print(
        "Chunking: OFF"
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Gradient accumulation:",
        GRAD_ACCUM_STEPS,
    )

    print(
        "Effective batch:",
        BATCH_SIZE
        * GRAD_ACCUM_STEPS,
    )


    # ========================================================
    # 10. 데이터
    #
    # Official Validation은 사용하지 않음
    # Training 내부 train/dev만 사용
    # ========================================================

    train_df = (
        pd.read_csv(
            TRAIN_PATH
        )
    )

    dev_df = (
        pd.read_csv(
            DEV_PATH
        )
    )


    if args.smoke:

        print(
            "\n"
            "*** SMOKE TEST MODE ***"
        )

        train_df = (
            train_df
            .iloc[:256]
            .copy()
        )

        dev_df = (
            dev_df
            .iloc[:256]
            .copy()
        )

        epochs = 1

    else:

        epochs = EPOCHS


    print(
        "\nTrain:",
        len(train_df),
    )

    print(
        "Dev:",
        len(dev_df),
    )


    # ========================================================
    # 11. Tokenizer
    # ========================================================

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_PATH,

            local_files_only=True,
        )
    )


    # ========================================================
    # 12. Dataset
    # ========================================================

    train_dataset = (
        Mission3TruncationDataset(
            train_df,

            tokenizer,

            desc=(
                "Train tokenizing"
            ),
        )
    )

    dev_dataset = (
        Mission3TruncationDataset(
            dev_df,

            tokenizer,

            desc=(
                "Dev tokenizing"
            ),
        )
    )


    # ========================================================
    # 13. DataLoader
    # ========================================================

    collator = (
        TruncationCollator(
            tokenizer
        )
    )

    train_loader = (
        DataLoader(
            train_dataset,

            batch_size=(
                BATCH_SIZE
            ),

            shuffle=True,

            collate_fn=(
                collator
            ),

            num_workers=0,

            pin_memory=True,
        )
    )

    dev_loader = (
        DataLoader(
            dev_dataset,

            batch_size=(
                BATCH_SIZE
            ),

            shuffle=False,

            collate_fn=(
                collator
            ),

            num_workers=0,

            pin_memory=True,
        )
    )


    # ========================================================
    # 14. Model
    # ========================================================

    model = (
        TruncationBaselineModel(
            MODEL_PATH
        )
    )

    model.to(
        device
    )


    # ========================================================
    # 15. Parameter 수
    # ========================================================

    total_params = sum(
        p.numel()
        for p
        in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p
        in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTotal parameters:",
        f"{total_params:,}",
    )

    print(
        "Active / Trainable parameters:",
        f"{trainable_params:,}",
    )


    # ========================================================
    # 16. Loss
    #
    # 04/06/07과 동일한 기본 BCE
    # ========================================================

    loss_fn = (
        nn.BCEWithLogitsLoss()
    )


    # ========================================================
    # 17. Optimizer
    # ========================================================

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),

            lr=(
                LEARNING_RATE
            ),

            weight_decay=(
                WEIGHT_DECAY
            ),
        )
    )


    # ========================================================
    # 18. Scheduler
    # ========================================================

    updates_per_epoch = (
        len(train_loader)
        + GRAD_ACCUM_STEPS
        - 1
    ) // GRAD_ACCUM_STEPS

    total_steps = (
        updates_per_epoch
        * epochs
    )

    warmup_steps = int(
        total_steps
        * WARMUP_RATIO
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,

            num_warmup_steps=(
                warmup_steps
            ),

            num_training_steps=(
                total_steps
            ),
        )
    )


    # ========================================================
    # 19. Mixed Precision
    # ========================================================

    scaler = (
        torch.amp.GradScaler(
            "cuda"
        )
    )


    # ========================================================
    # 20. Training
    # ========================================================

    best_macro_f1 = -1.0

    start_total = (
        time.time()
    )


    for epoch in range(
        1,
        epochs + 1,
    ):

        print(
            "\n"
        )

        print(
            "=" * 70
        )

        print(
            f"Epoch "
            f"{epoch}/"
            f"{epochs}"
        )

        print(
            "=" * 70
        )


        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0

        epoch_start = (
            time.time()
        )


        progress = tqdm(
            train_loader,

            desc=(
                f"Train E{epoch}"
            ),
        )


        for (
            step,
            batch,
        ) in enumerate(
            progress,
            start=1,
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

            labels = (
                batch[
                    "labels"
                ]
                .to(
                    device,

                    non_blocking=True,
                )
            )


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

                loss = (
                    loss_fn(
                        logits,
                        labels,
                    )
                )

                loss_for_backward = (
                    loss
                    / GRAD_ACCUM_STEPS
                )


            scaler.scale(
                loss_for_backward
            ).backward()


            running_loss += (
                loss.item()
            )


            if (
                step
                % GRAD_ACCUM_STEPS
                == 0

                or

                step
                == len(
                    train_loader
                )
            ):

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )


            progress.set_postfix(
                loss=(
                    running_loss
                    / step
                )
            )


        # ====================================================
        # 21. Internal Dev
        # ====================================================

        metrics = (
            evaluate(
                model,
                dev_loader,
                device,
                loss_fn,
            )
        )


        elapsed = (
            time.time()
            - epoch_start
        )


        print(
            f"\nEpoch time: "
            f"{elapsed / 60:.2f} min"
        )


        print(
            f"Dev loss: "
            f"{metrics['loss']:.6f}"
        )


        print(
            f"Dev Macro F1 "
            f"(threshold=0.5): "
            f"{metrics['macro_f1']:.6f}"
        )


        print(
            "\n"
            "Per-label metrics"
        )


        print(
            f"{'Label':8s} "
            f"{'P':>8s} "
            f"{'R':>8s} "
            f"{'F1':>8s}"
        )


        for (
            i,
            label,
        ) in enumerate(
            TARGET_SYMPTOMS
        ):

            print(
                f"{label:8s} "
                f"{metrics['precision'][i]:8.4f} "
                f"{metrics['recall'][i]:8.4f} "
                f"{metrics['f1'][i]:8.4f}"
            )


        # ====================================================
        # 22. Best model
        # ====================================================

        if (
            metrics[
                "macro_f1"
            ]
            > best_macro_f1
        ):

            best_macro_f1 = (
                metrics[
                    "macro_f1"
                ]
            )


            if not args.smoke:

                checkpoint = {

                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "macro_f1":
                        best_macro_f1,

                    "target_symptoms":
                        TARGET_SYMPTOMS,

                    "max_length":
                        MAX_LENGTH,

                    "threshold":
                        THRESHOLD,

                    "chunking":
                        False,

                    "pooling":
                        "none_cls",

                    "loss":
                        "BCEWithLogitsLoss",

                    "model_name":
                        "klue-roberta-base",
                }


                save_path = (
                    OUTPUT_DIR
                    / "best_model.pt"
                )


                torch.save(
                    checkpoint,
                    save_path,
                )


                print(
                    "\n"
                    "Best model saved:"
                )

                print(
                    save_path
                )


    # ========================================================
    # 23. 완료
    # ========================================================

    total_elapsed = (
        time.time()
        - start_total
    )


    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "Training completed"
    )

    print(
        "=" * 70
    )


    print(
        f"Best Dev Macro F1: "
        f"{best_macro_f1:.6f}"
    )


    print(
        f"Decision Threshold: "
        f"{THRESHOLD}"
    )


    print(
        f"Total time: "
        f"{total_elapsed / 60:.2f} min"
    )


if __name__ == "__main__":

    main()
