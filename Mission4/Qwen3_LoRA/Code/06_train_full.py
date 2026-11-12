
# ============================================================
# DCC Mission 4 - Qwen3-4B QLoRA DDP Training
# 06_train_full.py
# 8 GPU, gradient accumulation, checkpoints, logs
# ============================================================

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    PeftModel,
)

ROOT = Path("/data1/workspaces/ddc/DCC_Final")
WORK = ROOT / "Work/Qwen3_LoRA"
MODEL_DIR = WORK / "Models/Qwen3-4B"
DATA_FILE = WORK / "Data/prefix_train_8.jsonl"

LABELS = [
    "고열", "구토", "두통", "복통", "어지러움",
    "열상", "오심", "전신쇠약", "호흡곤란",
]


class PrefixDataset(Dataset):
    def __init__(self, path, limit=0):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
                if limit and len(self.rows) >= limit:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def create_collator(tokenizer, max_length):
    def collate(rows):
        texts = [r["text"] for r in rows]

        batch = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(
            [r["label_vector"] for r in rows],
            dtype=torch.float32,
        )

        return batch

    return collate


def build_model(local_rank, resume_adapter=None):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        num_labels=9,
        problem_type="multi_label_classification",
        quantization_config=quant_config,
        device_map={"": local_rank},
        dtype=torch.float16,
        attn_implementation="sdpa",
    )

    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
    )

    if resume_adapter:
        model = PeftModel.from_pretrained(
            model,
            resume_adapter,
            is_trainable=True,
        )
    else:
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type=TaskType.SEQ_CLS,
            bias="none",
            modules_to_save=["score"],
        )
        model = get_peft_model(model, config)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    model.enable_input_require_grads()
    model.train()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--run-name", type=str, default="ddp_test")
    parser.add_argument("--resume-adapter", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    assert args.batch_size >= 1
    assert args.grad_accum >= 1
    assert args.epochs >= 1
    assert args.limit >= 0
    assert args.max_length >= 1

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if distributed:
        dist.init_process_group(backend="nccl")

    try:
        seed = args.seed + rank
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if rank == 0:
            print("=" * 60, flush=True)
            print("Qwen3-4B QLoRA Distributed Training", flush=True)
            print("=" * 60, flush=True)
            print("GPU count:", world_size, flush=True)
            print("Train limit:", args.limit or "ALL", flush=True)
            print("Epochs:", args.epochs, flush=True)
            print("Max length:", args.max_length, flush=True)
            print(
                "Effective batch:",
                args.batch_size * args.grad_accum * world_size,
                flush=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 긴 대화에서는 최신 발화를 보존
        tokenizer.truncation_side = "right"

        dataset = PrefixDataset(DATA_FILE, args.limit)

        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
        else:
            sampler = None

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=0,
            collate_fn=create_collator(
                tokenizer, args.max_length
            ),
            pin_memory=True,
        )

        model = build_model(local_rank, args.resume_adapter)

        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        raw_model = model.module if distributed else model

        if rank == 0:
            raw_model.print_trainable_parameters()

        params = [
            p for p in raw_model.parameters()
            if p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=0.01,
        )

        output_dir = WORK / "Checkpoints" / args.run_name

        if rank == 0:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise RuntimeError(
                    f"출력 폴더가 이미 존재하고 비어 있지 않음: {output_dir}"
                )
            output_dir.mkdir(parents=True, exist_ok=True)

        if distributed:
            dist.barrier()

        start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            total_loss = 0.0
            total_batches = len(loader)
            optimizer_steps = 0

            for step, batch in enumerate(loader, start=1):
                batch = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                }

                labels = batch.pop("labels")

                group_start = (
                    (step - 1) // args.grad_accum
                ) * args.grad_accum

                group_size = min(
                    args.grad_accum,
                    total_batches - group_start,
                )

                should_update = (
                    step % args.grad_accum == 0
                    or step == total_batches
                )

                context = (
                    model.no_sync()
                    if distributed and not should_update
                    else nullcontext()
                )

                with context:
                    outputs = model(**batch)

                    loss = F.binary_cross_entropy_with_logits(
                        outputs.logits.float(),
                        labels,
                    )

                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite loss: {loss.item()}"
                        )

                    (loss / group_size).backward()

                total_loss += loss.item()

                if should_update:
                    torch.nn.utils.clip_grad_norm_(
                        params, max_norm=1.0
                    )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

                if rank == 0 and (
                    step == 1
                    or step % 100 == 0
                    or step == total_batches
                ):
                    mem = (
                        torch.cuda.max_memory_reserved(local_rank)
                        / 1024**3
                    )

                    print(
                        f"[Epoch {epoch+1}] "
                        f"step={step}/{total_batches} "
                        f"loss={loss.item():.4f} "
                        f"avg={total_loss/step:.4f} "
                        f"peak_mem={mem:.2f}GiB "
                        f"updates={optimizer_steps}",
                        flush=True,
                    )

            stats = torch.tensor(
                [total_loss, total_batches],
                dtype=torch.float64,
                device=device,
            )

            if distributed:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            avg_loss = (stats[0] / stats[1]).item()

            if rank == 0:
                epoch_dir = output_dir / f"epoch_{epoch+1}"
                epoch_dir.mkdir(parents=True, exist_ok=True)

                raw_model.save_pretrained(str(epoch_dir))
                tokenizer.save_pretrained(str(epoch_dir))

                with open(
                    epoch_dir / "training_state.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump({
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "world_size": world_size,
                        "max_length": args.max_length,
                        "effective_batch": (
                            args.batch_size
                            * args.grad_accum
                            * world_size
                        ),
                        "elapsed_seconds": time.time() - start,
                    }, f, ensure_ascii=False, indent=2)

                print(
                    f"[Epoch {epoch+1} 완료] "
                    f"mean_loss={avg_loss:.4f} "
                    f"saved={epoch_dir}",
                    flush=True,
                )

            if distributed:
                dist.barrier()

        if rank == 0:
            print("TRAINING COMPLETED", flush=True)

    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

