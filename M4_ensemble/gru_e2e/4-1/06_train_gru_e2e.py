# ============================================================
# 팀원3(Hierarchical Transformer + GRU) - end-to-end 미세조정
#
# 04_train_gru.py(frozen)와의 차이: klue-roberta-base를 더 이상
# 고정된 "번역기"로만 쓰지 않고, GRU/분류head와 같이 학습시킨다.
# 그래서 03_extract_utt_embeddings.py가 만든 임베딩 캐시를 안 쓰고,
# mission4_{train,val}.jsonl의 원문 텍스트를 매 배치마다 직접
# 토크나이즈 -> 인코딩(gradient 흐름) -> GRU로 흘려보낸다.
#
# 배치 구성 방식
#   배치 안의 여러 통화에 있는 발화 전부를 하나로 평평하게 펼쳐서
#   (통화 경계 무시) klue-roberta에 한 번에 통과시킨다. 그래야 발화
#   하나씩 따로 인코딩할 때보다 GPU를 훨씬 효율적으로 쓸 수 있다.
#   인코딩 후에는 (통화, 발화위치) 매핑대로 다시 (B, Lmax, hidden)
#   모양으로 되돌려서 GRU에 넣는다.
#
# 학습률: encoder(klue-roberta)와 GRU/분류head에 서로 다른 lr을 쓴다.
#   사전학습된 encoder를 GRU와 같은 큰 lr로 학습시키면 사전학습
#   지식이 금방 망가진다(catastrophic forgetting) - 이게 미세조정에서
#   가장 흔한 실수다.
#
# 체크포인트: 이제 encoder 가중치도 더 이상 공개된 klue-roberta-base와
# 같지 않으므로(미세조정됐으므로), 체크포인트에 encoder까지 전부
# 저장한다. 04_train_gru.py의 체크포인트(encoder는 공개 가중치를
# 그대로 다시 받아오는 방식)와는 로딩 방식이 다르다.
# ============================================================

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

from text_clean import clean_for_model

SYMPTOM_CLASSES = [
    "고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란",
]

SPEAKER_VOCAB = 4
SPEAKER_PAD_IDX = 3


def map_speaker(s):
    if s == 0:
        return 0
    if s == 1:
        return 1
    return 2


# ------------------------------------------------------------
# Dataset - 원문 텍스트를 그대로 들고 있다가 collate에서 평평하게 편다
# ------------------------------------------------------------

class CallTextDataset(Dataset):
    def __init__(self, jsonl_path):
        self.calls = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.calls.append(json.loads(line))

    def __len__(self):
        return len(self.calls)

    def __getitem__(self, idx):
        call = self.calls[idx]
        texts = [clean_for_model(u["text"]) for u in call["utterances"]]
        speakers = [map_speaker(u["speaker"]) for u in call["utterances"]]
        label_vector = call["label_vector"]
        return texts, speakers, label_vector, len(texts)


def collate_text(batch):
    lengths = torch.tensor([b[3] for b in batch], dtype=torch.int64)
    max_len = int(lengths.max())
    B = len(batch)

    speaker_padded = torch.full((B, max_len), SPEAKER_PAD_IDX, dtype=torch.int64)
    labels = torch.zeros(B, len(SYMPTOM_CLASSES), dtype=torch.float32)

    flat_texts = []
    batch_idx, pos_idx = [], []
    for i, (texts, speakers, label_vector, length) in enumerate(batch):
        speaker_padded[i, :length] = torch.tensor(speakers, dtype=torch.int64)
        labels[i] = torch.tensor(label_vector, dtype=torch.float32)
        for pos, t in enumerate(texts):
            flat_texts.append(t if t else "")
            batch_idx.append(i)
            pos_idx.append(pos)

    return (flat_texts, torch.tensor(batch_idx), torch.tensor(pos_idx),
            speaker_padded, labels, lengths, max_len)


# ------------------------------------------------------------
# 모델 - encoder까지 학습 대상에 포함
# ------------------------------------------------------------

class HierE2EModel(nn.Module):
    def __init__(self, backbone="klue/roberta-base", speaker_dim=16, hidden=256,
                 num_layers=1, num_classes=len(SYMPTOM_CLASSES), dropout=0.2,
                 max_length=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone)
        self.emb_dim = self.encoder.config.hidden_size
        self.max_length = max_length

        self.speaker_emb = nn.Embedding(SPEAKER_VOCAB, speaker_dim, padding_idx=SPEAKER_PAD_IDX)
        # +1: 지금까지 들은 발화 수(step feature). 전체 길이가 아니라
        # "현재까지의 위치"만 쓰므로 규정 위반이 아니다 (utt.index와 동일).
        self.gru = nn.GRU(
            input_size=self.emb_dim + speaker_dim + 1, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        self.hidden = hidden
        self.num_layers = num_layers

    def encode_flat_texts(self, flat_texts, device):
        enc = self.tokenizer(
            flat_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(device)
        out = self.encoder(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return pooled  # (N_total, emb_dim), gradient 흐름

    def forward(self, flat_texts, batch_idx, pos_idx, speaker_padded, lengths, max_len, device):
        pooled = self.encode_flat_texts(flat_texts, device)  # (N_total, emb_dim)

        B = speaker_padded.size(0)
        emb_padded = torch.zeros(B, max_len, self.emb_dim, device=device, dtype=pooled.dtype)
        emb_padded[batch_idx.to(device), pos_idx.to(device)] = pooled

        spk = self.speaker_emb(speaker_padded.to(device))

        # step feature: 패딩 안의 위치 t는 항상 "그 통화의 t번째 발화"를
        # 뜻하므로(스캐터로 채워졌으므로), 배치 전체에 arange를 그대로 쓰면 된다.
        step_idx = torch.arange(max_len, device=device, dtype=pooled.dtype).unsqueeze(0).expand(B, max_len)
        step_feat = torch.log1p(step_idx).unsqueeze(-1)  # (B, max_len, 1)

        x = torch.cat([emb_padded, spk, step_feat], dim=-1)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=max_len)

        logits = self.classifier(self.dropout(out))
        return logits


def compute_pos_weight(jsonl_path, cap=5.0):
    """
    클래스별 pos_weight = min(음성 수/양성 수, cap). cap을 안 씌우면
    희귀 클래스(두통 등)의 weight가 9배 넘게 나와서 학습이 불안정해질
    수 있어 상한을 둔다.
    """
    counts = [0] * len(SYMPTOM_CLASSES)
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for i, v in enumerate(d["label_vector"]):
                counts[i] += v
            n += 1
    weights = []
    for c in counts:
        w = (n - c) / max(c, 1)
        weights.append(min(w, cap))
    return torch.tensor(weights, dtype=torch.float32), counts, n


def compute_loss(logits, labels, lengths, device, pos_weight=None):
    B, L, C = logits.shape
    mask = (torch.arange(L, device=device)[None, :] < lengths[:, None].to(device))

    targets = labels.to(device).unsqueeze(1).expand(B, L, C)
    pw = pos_weight.to(device) if pos_weight is not None else None
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
    bce = bce.mean(dim=2) * mask.float()

    per_call = bce.sum(dim=1) / lengths.to(device).float()
    return per_call.mean()


# ------------------------------------------------------------
# 학습 루프
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--backbone", default="klue/roberta-base")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--speaker_dim", type=int, default=16)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8, help="통화 단위 배치 크기 (발화는 자동으로 훨씬 많아짐)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr_encoder", type=float, default=2e-5, help="klue-roberta용 학습률 (작게)")
    ap.add_argument("--lr_head", type=float, default=1e-3, help="GRU/분류head용 학습률 (크게)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1,
                     help="전체 스텝 중 웜업 비율 - encoder가 랜덤 초기화된 head의 "
                          "노이즈 섞인 gradient에 초반부터 흔들리는 걸 막기 위함")
    ap.add_argument("--pos_weight_cap", type=float, default=5.0, help="클래스 불균형 보정 상한")
    ap.add_argument("--resume_from", default=None,
                     help="이 체크포인트의 모델 가중치로 시작해서 --epochs만큼 이어서 학습. "
                          "옵티마이저 상태는 복원 안 됨(새로 시작) - 재개할 때는 보통 "
                          "--warmup_ratio 0도 같이 주는 걸 추천 (encoder가 이미 적응된 "
                          "상태라 다시 웜업할 필요 없음).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_fp16", action="store_true", help="fp16 autocast 끄기 (디버깅용)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[info] device = {device}")

    train_ds = CallTextDataset(args.train_jsonl)
    val_ds = CallTextDataset(args.val_jsonl)
    print(f"[info] train {len(train_ds)}통화 / val {len(val_ds)}통화")

    pos_weight, pos_counts, n_train = compute_pos_weight(args.train_jsonl, cap=args.pos_weight_cap)
    print("[info] pos_weight (cap={:.1f}):".format(args.pos_weight_cap))
    for c, cnt, w in zip(SYMPTOM_CLASSES, pos_counts, pos_weight.tolist()):
        print(f"    {c:6s} 양성 {cnt:5d}/{n_train}  weight={w:.2f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_text)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_text)

    model = HierE2EModel(
        backbone=args.backbone, speaker_dim=args.speaker_dim, hidden=args.hidden,
        num_layers=args.num_layers, dropout=args.dropout, max_length=args.max_length,
    ).to(device)

    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(resume_ckpt["model_state"])
        print(f"[info] {args.resume_from} 에서 가중치 복원 완료 - 이어서 {args.epochs}에폭 학습")

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr_encoder},
        {"params": list(model.speaker_emb.parameters()) + list(model.gru.parameters())
                    + list(model.classifier.parameters()), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    # 웜업은 encoder 전용이어야 한다. LambdaLR에 함수를 하나만 넘기면
    # optimizer의 모든 param group(= encoder, head 둘 다)에 똑같이
    # 적용되어 버린다 - 그러면 랜덤 초기화된 head까지 초반에 느리게
    # 배우게 되어 오히려 손해다. param group별로 다른 lambda를 줘서
    # encoder만 웜업하고, head는 처음부터 전속력 + 코사인 감소만 적용한다.
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def cosine_after(step, start):
        progress = (step - start) / max(1, total_steps - start)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    def encoder_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return cosine_after(step, warmup_steps)

    def head_lambda(step):
        return cosine_after(step, 0)   # 웜업 없이 처음부터 전속력, 코사인 감소만

    # optimizer 선언 순서가 [encoder, head]였으므로 이 순서를 그대로 맞춘다.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[encoder_lambda, head_lambda]
    )
    print(f"[info] encoder만 웜업 {warmup_steps}/{total_steps} 스텝, head는 웜업 없음")

    use_fp16 = (device.type == "cuda") and not args.no_fp16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    print(f"[info] fp16 autocast = {use_fp16}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        skipped_steps = 0
        grad_norm_sum = 0.0

        for flat_texts, batch_idx, pos_idx, speaker_padded, labels, lengths, max_len in train_loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                logits = model(flat_texts, batch_idx, pos_idx, speaker_padded, lengths, max_len, device)
                loss = compute_loss(logits, labels, lengths, device, pos_weight=pos_weight)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm_sum += float(total_norm)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped_steps += 1   # GradScaler가 inf/nan 감지하고 이 스텝을 건너뜀
            scheduler.step()   # 웜업+코사인은 배치마다 갱신해야 함 (에폭마다 X)

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        train_elapsed = time.time() - t0
        avg_grad_norm = grad_norm_sum / n_batches
        print(f"  [diag] skipped_steps={skipped_steps}/{n_batches}  "
              f"clip 전 평균 grad_norm={avg_grad_norm:.2f}  scaler_scale={scaler.get_scale():.1f}")

        model.eval()
        val_loss_total, val_batches = 0.0, 0
        with torch.no_grad():
            for flat_texts, batch_idx, pos_idx, speaker_padded, labels, lengths, max_len in val_loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                    logits = model(flat_texts, batch_idx, pos_idx, speaker_padded, lengths, max_len, device)
                    val_loss_total += compute_loss(logits, labels, lengths, device, pos_weight=pos_weight).item()
                val_batches += 1
        val_loss = val_loss_total / val_batches
        epoch_elapsed = time.time() - t0

        cur_lr_enc = optimizer.param_groups[0]["lr"]
        cur_lr_head = optimizer.param_groups[1]["lr"]
        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  lr_encoder={cur_lr_enc:.2e}  lr_head={cur_lr_head:.2e}  "
              f"({epoch_elapsed:.0f}s, train {train_elapsed:.0f}s)")

        ckpt = {
            "model_state": model.state_dict(),   # encoder 포함 전체
            "config": {
                "backbone": args.backbone, "speaker_dim": args.speaker_dim,
                "hidden": args.hidden, "num_layers": args.num_layers,
                "dropout": args.dropout, "max_length": args.max_length,
                "symptom_classes": SYMPTOM_CLASSES, "e2e": True,
            },
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> best.pt 갱신 (val_loss={val_loss:.4f})")

    print(f"\n[done] 저장 위치: {out_dir}  (best.pt / last.pt)")


if __name__ == "__main__":
    main()
