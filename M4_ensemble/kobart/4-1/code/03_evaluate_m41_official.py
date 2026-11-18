import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from m4_model import load_checkpoint
ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / 'Work' / 'Common'
DATA = ROOT / 'Work' / 'Seq2Seq'
with open(COMMON / 'label_map.json', encoding='utf-8') as f:
    LABELS = json.load(f)['labels']
assert len(LABELS) == 9

def batch_reader(path, batch_size):
    batch = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

def moment_score(gold, pred):
    union = gold | pred
    if not union:
        return 1.0
    return len(gold & pred) / len(union)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['kobart', 't5'], required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    ckpt = ROOT / 'Checkpoints' / args.model / args.experiment
    if not (ckpt / 'model.pt').exists():
        raise FileNotFoundError(f'체크포인트 없음: {ckpt}')
    with open(ckpt / 'metadata.json', encoding='utf-8') as f:
        metadata = json.load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device, flush=True)
    print('Checkpoint:', ckpt, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    tokenizer.truncation_side = 'left'
    model = load_checkpoint(ckpt, device)
    model.eval()
    input_path = ROOT / 'Work' / 'OfficialVal' / 'official_val_t5.jsonl'
    out_dir = ROOT / 'Results' / 'M4_1'
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{args.model}_{args.experiment}_{args.split}'
    pred_path = out_dir / f'{tag}.csv'
    prob_path = out_dir / f'{tag}_probs.jsonl'
    metric_path = out_dir / f'{tag}_metrics.json'
    for path in [pred_path, prob_path, metric_path]:
        if path.exists():
            raise FileExistsError(f'기존 결과 보호: {path}')
    call_scores = defaultdict(float)
    call_counts = defaultdict(int)
    last_steps = defaultdict(int)
    total = 0
    with pred_path.open('w', encoding='utf-8', newline='') as pred_f, prob_path.open('w', encoding='utf-8') as prob_f:
        writer = csv.writer(pred_f)
        writer.writerow(['label_file', 'step', 'symptom'])
        for batch in batch_reader(input_path, args.batch):
            texts = [r['input_text'] for r in batch]
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=metadata['max_input'], return_tensors='pt').to(device)
            with torch.inference_mode():
                probs = model.predict_probs(inputs['input_ids'], inputs['attention_mask'])
            if not torch.isfinite(probs).all():
                raise ValueError('NaN/Inf 예측 확률 발견')
            probs = probs.cpu().tolist()
            for row, p in zip(batch, probs):
                lf = row['label_file']
                step = int(row['step'])
                if args.split == 'val':
                    expected = last_steps[lf] + 1
                    if step != expected:
                        raise ValueError(f'{lf}: step={step}, expected={expected}')
                    last_steps[lf] = step
                pred_indices = {i for i in range(9) if p[i] >= 0.5}
                prediction = [LABELS[i] for i in range(9) if i in pred_indices]
                writer.writerow([lf, step, ';'.join(prediction)])
                prob_f.write(json.dumps({'label_file': lf, 'step': step, 'probs': p}, ensure_ascii=False) + '\n')
                total += 1
            if total % 5000 < len(batch):
                print(f'[진행] {total:,} prefixes', flush=True)
    if total != 111947:
        raise ValueError(f'Prefix 개수 불일치: {total}')
    if len(last_steps) != 3640:
        raise ValueError(f'통화 개수 불일치: {len(last_steps)}')
    results = {'model': args.model, 'experiment': args.experiment, 'split': 'official_val', 'calls': len(last_steps), 'prefixes': total, 'inference_only': True}
    with metric_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('===== 공식 Validation 추론 완료 =====', flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
