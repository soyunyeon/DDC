import importlib.util
import json
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path("/data1/workspaces/ddc/DCC_Final")
WORK = ROOT / "Work/Qwen3_LoRA"

spec = importlib.util.spec_from_file_location(
    "qwen_model",
    WORK / "Code/04_model.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

tokenizer = module.load_tokenizer()
model = module.build_model()

model.train()
device = torch.device("cuda:0")

# 실제 1024토큰 이상인 Prefix 하나 선택
path = WORK / "Data/prefix_long_test.jsonl"
with path.open(encoding="utf-8") as f:
    row = json.loads(next(f))

tokenizer.truncation_side = "right"

batch = tokenizer(
    row["text"],
    truncation=True,
    max_length=1024,
    return_tensors="pt",
).to(device)

labels = torch.tensor(
    [row["label_vector"]],
    dtype=torch.float32,
    device=device,
)

print("실제 입력 토큰 수:", batch["input_ids"].shape[1], flush=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
)

optimizer.zero_grad(set_to_none=True)

outputs = model(**batch)
loss = F.binary_cross_entropy_with_logits(
    outputs.logits.float(),
    labels,
)

print("Forward Loss:", loss.item(), flush=True)

loss.backward()
print("Backward 성공", flush=True)

optimizer.step()
print("Optimizer Step 성공", flush=True)

print(
    "GPU 최대 예약 메모리:",
    round(torch.cuda.max_memory_reserved(0) / 1024**3, 2),
    "GiB",
    flush=True,
)

print("LONG PREFIX TEST COMPLETED", flush=True)
