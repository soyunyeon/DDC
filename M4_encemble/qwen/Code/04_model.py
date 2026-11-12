# ============================================================
# DCC Final Mission 4
# 04_model.py
#
# Qwen3-4B + 4bit QLoRA + Multi-label Classification
# Output: 9 symptom logits
# ============================================================

import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


ROOT = Path("/data1/workspaces/ddc/DCC_Final")

MODEL_PATH = (
    ROOT / "Work/Qwen3_LoRA/Models/Qwen3-4B"
)

SYMPTOMS = [
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

NUM_LABELS = len(SYMPTOMS)

MAX_LENGTH = 1024

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ------------------------------------------------------------
# 1. Tokenizer
# ------------------------------------------------------------

def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH)
    )

    # Qwen3는 별도 pad token이 없을 수 있으므로
    # 기존 EOS 토큰을 padding에 사용
    tokenizer.pad_token = tokenizer.eos_token

    # 마지막 유효 토큰으로 분류하기 쉽도록 우측 padding
    tokenizer.padding_side = "right"

    return tokenizer


# ------------------------------------------------------------
# 2. Qwen3 + QLoRA
# ------------------------------------------------------------

def build_model():

    print("[1] 4bit 양자화 설정", flush=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("[2] Qwen3-4B 분류 모델 로딩", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH),

        num_labels=NUM_LABELS,

        problem_type="multi_label_classification",

        quantization_config=quant_config,

        device_map={"": 0},

        dtype=torch.float16,

        attn_implementation="sdpa",
    )

    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False

    # QLoRA 학습 준비
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    print("[3] LoRA 설정", flush=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

        # 분류 모델용 PEFT Wrapper
        task_type=TaskType.SEQ_CLS,

        bias="none",

        # 분류 헤드도 학습·저장
        modules_to_save=["score"],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    print("[4] 학습 가능 파라미터", flush=True)

    model.print_trainable_parameters()

    return model


# ------------------------------------------------------------
# 3. 단독 모델 테스트
# ------------------------------------------------------------

def test_model():

    print("=" * 60)
    print("Qwen3-4B QLoRA Classification Test")
    print("=" * 60)

    tokenizer = load_tokenizer()
    model = build_model()

    text = (
        "119대원: 119입니다. 무슨 일이세요?\n"
        "신고자: 배가 너무 아프고 계속 토해요."
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        k: v.to("cuda:0")
        for k, v in inputs.items()
    }

    # 정답 예시: 구토 + 복통
    labels = torch.tensor(
        [[0, 1, 0, 1, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
        device="cuda:0",
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            **inputs,
            labels=labels,
        )

        logits = outputs.logits

        probabilities = torch.sigmoid(
            logits.float()
        )

    print("\n[5] Forward Pass 결과")

    print("Logits shape:", tuple(logits.shape))

    print("BCE Loss:", outputs.loss.item())

    print("\n증상별 확률 (학습 전)")

    for name, prob in zip(
        SYMPTOMS,
        probabilities[0].tolist()
    ):
        print(f"{name:8s}: {prob:.4f}")

    print("\n[6] GPU 메모리")

    print(
        "Allocated:",
        round(
            torch.cuda.memory_allocated(0) / 1024**3,
            2
        ),
        "GiB"
    )

    print(
        "Peak reserved:",
        round(
            torch.cuda.max_memory_reserved(0) / 1024**3,
            2
        ),
        "GiB"
    )

    print("\n모델 구조 테스트 완료")


if __name__ == "__main__":
    test_model()
