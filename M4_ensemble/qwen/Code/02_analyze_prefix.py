import json
import random
import statistics
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path("/data1/workspaces/ddc/DCC_Final")
DATA = ROOT / "Work/Common/mission4_train.jsonl"
MODEL = ROOT / "Work/Qwen3_LoRA/Models/Qwen3-4B"

random.seed(42)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

with DATA.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

samples = random.sample(rows, min(500, len(rows)))
counts = []
lengths = {0.25: [], 0.5: [], 0.75: [], 1.0: []}

for row in samples:
    utts = row["utterances"]
    counts.append(len(utts))

    for ratio in lengths:
        k = max(1, int(len(utts) * ratio))
        text = "\n".join(
            ("신고자" if u["speaker"] == 1 else "119대원")
            + ": " + u["text"]
            for u in utts[:k]
        )
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        lengths[ratio].append(n_tokens)

print("분석 통화:", len(samples))
print("평균 발화 수:", round(statistics.mean(counts), 1))
print("최대 발화 수:", max(counts))

for ratio, values in lengths.items():
    values.sort()
    print(
        f"Prefix {ratio:.0%} | "
        f"평균 {statistics.mean(values):.0f} | "
        f"P95 {values[int(len(values)*0.95)]:d} | "
        f"최대 {max(values)}"
    )
