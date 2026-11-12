
# ============================================================
# DCC Mission 4 - Qwen3 QLoRA Training
# 05_train.py
# Smoke test: actual backward + optimizer + checkpoint
# ============================================================

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/data1/workspaces/ddc/DCC_Final")
WORK = ROOT / "Work/Qwen3_LoRA"
TRAIN_FILE = WORK / "Data/prefix_train_8.jsonl"
MODEL_FILE = WORK / "Code/04_model.py"


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "qwen3_m4_model", MODEL_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_samples(path, limit, seed=42):
    rng = random.Random(seed)
    samples = []

    # 전체 학습 데이터에서 무작위 표본 추출
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            if len(samples) < limit:
                samples.append(record)
            else:
                j = rng.randint(0, i)
                if j < limit:
                    samples[j] = record

    rng.shuffle(samples)

    for row in samples:
        assert len(row["label_vector"]) == 9
        assert 1 <= row["step"]

    return samples


class PrefixDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": row["label_vector"],
        }


def make_collate(tokenizer):
    def collate(batch):
        inputs = tokenizer.pad(
            [
                {
                    "input_ids": x["input_ids"],
                    "attention_mask": x["attention_mask"],
                }
                for x in batch
            ],
            padding=True,
            return_tensors="pt",
        )
        inputs["labels"] = torch.tensor(
            [x["labels"] for x in batch],
            dtype=torch.float32,
        )
        return inputs

    return collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert args.limit > 0
    assert args.grad_accum > 0
    assert args.max_length > 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    print("=" * 60, flush=True)
    print("Qwen3-4B QLoRA Smoke Training", flush=True)
    print("=" * 60, flush=True)

    print("[1] 학습 데이터 준비", flush=True)

    rows = load_samples(
        TRAIN_FILE, args.limit, args.seed
    )

    print("학습 Prefix:", len(rows), flush=True)
    print(
        "고유 통화:",
        len({r["call_id"] for r in rows}),
        flush=True,
    )

    print("[2] Qwen3 모델 로딩", flush=True)

    module = load_model_module()
    tokenizer = module.load_tokenizer()
    model = module.build_model()

    device = torch.device("cuda:0")
    model.train()

    dataset = PrefixDataset(
        rows, tokenizer, args.max_length
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate(tokenizer),
    )

    params = [
        p for p in model.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(loader)
    losses = []
    start = time.time()

    optimizer.zero_grad(set_to_none=True)

    print(
        f"[3] 학습 시작 | samples={total_steps} "
        f"| max_length={args.max_length} "
        f"| grad_accum={args.grad_accum}",
        flush=True,
    )

    accumulation_count = 0
    optimizer_steps = 0
    running_loss = 0.0

    for step, batch in enumerate(loader, start=1):

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        labels = batch.pop("labels")

        outputs = model(**batch)

        # FP32에서 BCE 계산해 수치 안정성 확보
        loss = F.binary_cross_entropy_with_logits(
            outputs.logits.float(),
            labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: {loss.item()}"
            )

        # 미니배치별 기울기 누적
        # 마지막 부분 배치는 별도 보정
        group_start = (
            (step - 1) // args.grad_accum
        ) * args.grad_accum

        group_size = min(
            args.grad_accum,
            total_steps - group_start,
        )

        (loss / group_size).backward()

        accumulation_count += 1

        loss_value = float(loss.item())
        losses.append(loss_value)
        running_loss += loss_value

        should_update = (
            accumulation_count == group_size
        )

        if should_update:

            torch.nn.utils.clip_grad_norm_(
                params, max_norm=1.0
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_steps += 1
            accumulation_count = 0

        if step == 1 or step % 20 == 0 or step == total_steps:
            avg = running_loss / (
                step % 20 if step % 20 else 20
            )

            if step == 1:
                avg = loss_value

            mem = (
                torch.cuda.max_memory_reserved(0)
                / 1024**3
            )

            print(
                f"[Train] {step:4d}/{total_steps} "
                f"loss={loss_value:.4f} "
                f"recent_avg={avg:.4f} "
                f"peak_mem={mem:.2f}GiB "
                f"updates={optimizer_steps}",
                flush=True,
            )

            running_loss = 0.0

    elapsed = time.time() - start

    print("\n[4] 학습 완료", flush=True)
    print("첫 20개 Loss 평균:",
          sum(losses[:20]) / len(losses[:20]))
    print("마지막 20개 Loss 평균:",
          sum(losses[-20:]) / len(losses[-20:]))
    print("Optimizer steps:", optimizer_steps)
    print("총 소요 시간(초):", round(elapsed, 1))
    print(
        "최대 GPU 예약 메모리(GiB):",
        round(
            torch.cuda.max_memory_reserved(0) / 1024**3,
            2,
        ),
    )

    print("[5] 체크포인트 저장", flush=True)

    save_dir = (
        WORK / "Checkpoints/smoke_200"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    metrics = {
        "limit": len(rows),
        "max_length": args.max_length,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "optimizer_steps": optimizer_steps,
        "first_20_loss": sum(losses[:20]) / len(losses[:20]),
        "last_20_loss": sum(losses[-20:]) / len(losses[-20:]),
        "elapsed_seconds": elapsed,
        "peak_memory_gib": (
            torch.cuda.max_memory_reserved(0) / 1024**3
        ),
    }

    with open(
        save_dir / "smoke_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    print("저장 경로:", save_dir, flush=True)
    print("Smoke Test 완료", flush=True)


if __name__ == "__main__":
    main()
