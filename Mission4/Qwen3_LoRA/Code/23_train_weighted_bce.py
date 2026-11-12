import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, TaskType, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

ROOT = Path('/data1/workspaces/ddc/DCC_Final')
WORK = ROOT / 'Work/Qwen3_LoRA'
MODEL_DIR = WORK / 'Models/Qwen3-4B'
TRAIN_FILE = WORK / 'Data/prefix_train_8.jsonl'
LABELS = ['고열', '구토', '두통', '복통', '어지러움', '열상', '오심', '전신쇠약', '호흡곤란']

class PrefixDataset(Dataset):
    def __init__(self, file, limit):
        with open(file, encoding='utf-8') as f:
            self.rows = [json.loads(line) for i, line in enumerate(f) if not limit or i < limit]
        assert self.rows and all(len(r['label_vector']) == 9 for r in self.rows)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.rows[i]

def collator(tokenizer, max_length):
    def fn(rows):
        # Preserve the earliest 1024 tokens; no future utterances are introduced.
        x = tokenizer([r['text'] for r in rows], padding=True, truncation=True,
                      max_length=max_length, return_tensors='pt')
        x['labels'] = torch.tensor([r['label_vector'] for r in rows], dtype=torch.float32)
        return x
    return fn

def build_model(gpu, adapter=None):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), num_labels=9, problem_type='multi_label_classification',
        quantization_config=quant, device_map={'': gpu}, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa')
    m.config.pad_token_id = m.config.eos_token_id
    m.config.use_cache = False
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
    if adapter:
        m = PeftModel.from_pretrained(m, str(adapter), is_trainable=True)
    else:
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                         'gate_proj', 'up_proj', 'down_proj'],
                         task_type=TaskType.SEQ_CLS, bias='none', modules_to_save=['score'])
        m = get_peft_model(m, cfg)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m.enable_input_require_grads()
    return m

def save_checkpoint(folder, model, tokenizer, optimizer, epoch, updates, step_in_epoch,
                    rank, world, args):
    folder.mkdir(parents=True, exist_ok=True) if rank == 0 else None
    if world > 1:
        dist.barrier()
    raw = model.module if world > 1 else model
    # Each rank stores its RNG to preserve dropout behavior after resume.
    torch.save({'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
                'python': random.getstate()}, folder / f'rng_rank{rank}.pt')
    if rank == 0:
        raw.save_pretrained(str(folder / 'adapter'))
        tokenizer.save_pretrained(str(folder / 'adapter'))
        torch.save(optimizer.state_dict(), folder / 'optimizer.pt')
        state = {'epoch': epoch, 'updates': updates, 'step_in_epoch': step_in_epoch,
                 'world_size': world, 'batch_size': args.batch_size,
                 'grad_accum': args.grad_accum, 'max_length': args.max_length,
                 'seed': args.seed, 'train_limit': args.limit}
        (folder / 'state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(f'[SAVE] {folder}', flush=True)
    if world > 1:
        dist.barrier()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--max-length', type=int, default=1024)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--save-every', type=int, default=500, help='optimizer steps')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', type=str, default=None, help='checkpoint directory')
    args = ap.parse_args()
    assert args.epochs > 0 and args.grad_accum > 0 and args.batch_size > 0
    assert args.save_every > 0 and args.limit >= 0 and args.max_length > 0
    rank, local_rank, world = (int(os.getenv(k, d)) for k, d in
                               [('RANK', '0'), ('LOCAL_RANK', '0'), ('WORLD_SIZE', '1')])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    try:
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'
        dataset = PrefixDataset(TRAIN_FILE, args.limit)
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                     shuffle=True, seed=args.seed) if world > 1 else None
        if sampler is None:
            # torchrun is used in the intended execution; single GPU fallback.
            loader_shuffle = True
        else:
            loader_shuffle = False
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            shuffle=loader_shuffle, num_workers=0,
                            collate_fn=collator(tokenizer, args.max_length))
        out = WORK / 'Checkpoints' / args.run_name
        resume_state = None
        if args.resume:
            ck = Path(args.resume)
            resume_state = json.loads((ck / 'state.json').read_text(encoding='utf-8'))
            for key, value in [('world_size', world), ('batch_size', args.batch_size),
                               ('grad_accum', args.grad_accum), ('max_length', args.max_length),
                               ('seed', args.seed), ('train_limit', args.limit)]:
                if resume_state[key] != value:
                    raise ValueError(f'Resume config mismatch: {key}')
            if out.resolve() != ck.parent.resolve():
                raise ValueError('Resume must use the same --run-name and checkpoint parent')
        elif out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Run directory not empty: {out}')
        model = build_model(local_rank, (Path(args.resume) / 'adapter') if args.resume else None)
        if world > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False, broadcast_buffers=False)
        raw = model.module if world > 1 else model
        params = [p for p in raw.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        start_epoch, resume_step, updates = 0, 0, 0
        if args.resume:
            optimizer.load_state_dict(torch.load(Path(args.resume) / 'optimizer.pt',
                                                 map_location=device, weights_only=False))
            for group in optimizer.param_groups:
                group['lr'] = args.lr
            start_epoch = resume_state['epoch']
            resume_step = resume_state['step_in_epoch']
            updates = resume_state['updates']
            # RNG is restored immediately before continuing the step loop.
        if rank == 0:
            print(f'[START] GPUs={world}, train={len(dataset)}, epochs={args.epochs}, '
                  f'max_length={args.max_length}, effective_batch={world*args.batch_size*args.grad_accum}, '
                  f'resume={args.resume}', flush=True)
            raw.print_trainable_parameters()
        optimizer.zero_grad(set_to_none=True)
        start = time.time()
        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            total = len(loader)
            epoch_loss = 0.0
            completed = resume_step if epoch == start_epoch else 0
            if args.resume and epoch == start_epoch:
                rng = torch.load(Path(args.resume) / f'rng_rank{rank}.pt',
                                 map_location='cpu', weights_only=False)
                # DataLoader iteration itself can consume global CPU torch RNG.
                # Restore saved states after skipped items have been iterated.
            for i, batch in enumerate(loader, start=1):
                if i <= completed:
                    continue
                if args.resume and epoch == start_epoch and i == completed + 1:
                    torch.set_rng_state(rng['torch'])
                    torch.cuda.set_rng_state(rng['cuda'], device=device)
                    random.setstate(rng['python'])
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop('labels')
                group_start = ((i - 1) // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, total - group_start)
                sync = (i % args.grad_accum == 0 or i == total)
                from contextlib import nullcontext
                with (model.no_sync() if world > 1 and not sync else nullcontext()):
                    y = model(**batch).logits.float()
                    loss = F.binary_cross_entropy_with_logits(
                        y,
                        labels,
                        pos_weight=torch.tensor(
                            [1.0, 1.3, 1.3, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0],
                            dtype=y.dtype,
                            device=y.device,
                        ),
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite loss at epoch={epoch+1}, step={i}')
                    (loss / group_size).backward()
                epoch_loss += float(loss.item())
                if sync:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    if updates % args.save_every == 0:
                        save_checkpoint(out / f'step_{updates:06d}', model, tokenizer,
                                        optimizer, epoch, updates, i, rank, world, args)
                if rank == 0 and (i == completed + 1 or i % 250 == 0 or i == total):
                    print(f'[EPOCH {epoch+1}] {i}/{total} loss={loss.item():.4f} '
                          f'peak={torch.cuda.max_memory_reserved(local_rank)/2**30:.2f}GiB '
                          f'updates={updates} elapsed={time.time()-start:.0f}s', flush=True)
            save_checkpoint(out / f'epoch_{epoch+1}', model, tokenizer, optimizer,
                            epoch+1, updates, 0, rank, world, args)
            resume_step = 0
        if rank == 0:
            print('[SUCCESS] TRAINING COMPLETED', flush=True)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()
