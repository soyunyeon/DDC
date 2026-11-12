import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_PATH = "Work/Qwen3_LoRA/Models/Qwen3-4B"

print("[1] Tokenizer 로딩", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("[2] 4bit Qwen3 모델 로딩", flush=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
)

print("[3] 모델 로딩 성공", flush=True)
print("GPU 메모리 할당:", round(torch.cuda.memory_allocated(0) / 1024**3, 2), "GiB")
print("GPU 최대 예약:", round(torch.cuda.max_memory_reserved(0) / 1024**3, 2), "GiB")

text = "119 신고자: 숨이 차고 호흡하기 어려워요."
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

with torch.inference_mode():
    outputs = model(**inputs)

print("[4] Forward Pass 성공", flush=True)
print("Logits Shape:", tuple(outputs.logits.shape))
print("Qwen3-4B 4bit 테스트 완료", flush=True)
