import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

ROOT = Path('/data1/workspaces/ddc/DCC_Final')
WORK = ROOT / 'Work/Qwen3_LoRA'
LABELS = ['고열', '구토', '두통', '복통', '어지러움', '열상', '오심', '전신쇠약', '호흡곤란']
MODEL_DIR = WORK / 'Models/Qwen3-4B'
ADAPTER_DIR = WORK / 'Checkpoints/qwen3_4b_qlora_v1/epoch_1/adapter'
VAL_FILE = WORK / 'Official_Val/official_val_prefix_all.jsonl'
OUT_DIR = WORK / 'Official_Val/Predictions'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=0, help='First N prefixes for quick smoke test; 0=all')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--output-tag', default='val_all')
    a = p.parse_args()
    assert a.limit >= 0 and a.batch_size > 0 and a.max_length > 0
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f'{a.output_tag}_rank{rank}.jsonl'
    token = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    token.pad_token = token.eos_token
    token.padding_side = 'right'
    token.truncation_side = 'right'
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), num_labels=9, problem_type='multi_label_classification',
        quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map={'': local_rank}, attn_implementation='sdpa', local_files_only=True)
    model.config.pad_token_id = token.pad_token_id
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), is_trainable=False,
                                    local_files_only=True)
    model.eval()

    completed = set()
    if output.exists():
        with output.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed.add((r['call_id'], int(r['step'])))
    print(f'[RESUME] rank={rank}, completed={len(completed)}', flush=True)
    pending = []
    seen = 0
    def infer(rows, out):
        enc = token([x['text'] for x in rows], padding=True, truncation=True,
                    max_length=a.max_length, return_tensors='pt').to(f'cuda:{local_rank}')
        with torch.inference_mode():
            probs = torch.sigmoid(model(**enc).logits.float()).cpu().tolist()
        for item, pr in zip(rows, probs):
            assert len(pr) == len(LABELS)
            if not all(0 <= v <= 1 for v in pr):
                raise ValueError(f'Invalid probabilities: {item["call_id"]} step={item["step"]}')
            out.write(json.dumps({'call_id': item['call_id'], 'step': item['step'],
                                  'label_file': item.get('label_file'),
                                  'label_vector': item['label_vector'],
                                  'probs': pr}, ensure_ascii=False) + '\n')

    with VAL_FILE.open(encoding='utf-8') as source, output.open('a', encoding='utf-8') as out:
        for idx, line in enumerate(source):
            if a.limit and idx >= a.limit:
                break
            if idx % world != rank:
                continue  # No DistributedSampler padding/duplicates.
            item = json.loads(line)
            if (item['call_id'], int(item['step'])) in completed:
                continue
            assert len(item['label_vector']) == 9
            pending.append(item)
            if len(pending) >= a.batch_size:
                infer(pending, out)
                out.flush()
                seen += len(pending)
                pending.clear()
                if seen % 2000 == 0:
                    print(f'[RANK {rank}] {seen} prefixes', flush=True)
        if pending:
            infer(pending, out)
            seen += len(pending)
    print(f'[DONE] rank={rank} rows={seen} file={output}', flush=True)


if __name__ == '__main__':
    main()
