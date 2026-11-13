# ============================================================
# 미션 4-2 전용 독립 모델 학습
#
# 06_train_gru_e2e.py와 뼈대(klue-roberta-large 발화 임베딩 + GRU
# 시퀀스 모델)는 완전히 같다. 다른 점은 딱 두 가지, 전부 "학습 방식"
# 레벨이고 결정 규칙(halt 판단 시 0.5+streak)은 전혀 안 건드린다 —
# 그래서 주최측 공지(결정 임계값 0.5 고정, 중단 판단 기준값도 0.5)와
# 무관하다. 4-1용 베스트 체크포인트(e2e_large)는 그대로 유지하고,
# 이 스크립트가 만드는 체크포인트는 4-2 전용으로만 쓴다.
#
# [레버 1] 랜덤 길이 자르기(truncation) 증강
#   06번은 항상 "통화 전체"로 학습해서, 모델이 "짧은 입력이 전부인
#   상황"을 학습 중엔 한 번도 직접 경험 못 한다(추론 때는 매번 일찍
#   끊기는데 학습 때는 항상 긴 시퀀스의 일부로만 등장). 매 학습 샘플을
#   trunc_prob 확률로 랜덤 길이만큼 잘라서 "그게 전부인 통화"처럼
#   학습에 섞어 넣는다 — GRU가 "여기서 끝"인 짧은 시퀀스에서도 안정된
#   판단을 내도록 직접 훈련시키는 게 목적.
#
# [레버 2] 안정성 정칙화(stability regularization)
#   val_holdout 진단 결과 통화의 82%가 중간에 "판정 집합이 줄어드는"
#   순간을 겪는다(한번 잡힌 증상이 확률이 흔들리며 다시 사라짐).
#   정답에 실제로 있는 클래스에 한해서만, 그 클래스의 확률이 직전
#   시점보다 (margin 이상) 떨어지면 작은 페널티를 준다(hinge). 전부
#   유지시키는 ratchet(결정 규칙으로 시도했다가 정밀도가 무너졌던
#   방식)과 달리, 정답에 없는 클래스에는 전혀 적용 안 되므로 거짓양성이
#   쌓이는 부작용이 없다 — "학습이 그렇게 가도록 유도"만 할 뿐, 최종
#   출력은 여전히 매 시점 독립적인 0.5 판정이다.
#
# 체크포인트 포맷은 06_train_gru_e2e.py와 100% 동일하게 저장한다
# (inference_gru_e2e.py가 그대로 로드 가능).
# ============================================================

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("train_gru_e2e_module", _THIS_DIR / "06_train_gru_e2e.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

HierE2EModel = _m.HierE2EModel
CallTextDataset = _m.CallTextDataset
collate_text = _m.collate_text
compute_pos_weight = _m.compute_pos_weight
map_speaker = _m.map_speaker
SYMPTOM_CLASSES = _m.SYMPTOM_CLASSES

from text_clean import clean_for_model


# ------------------------------------------------------------
# 레버 3(개선): 임베딩 풀링 선택 — masked-mean(06번 기본) vs CLS.
# 06_train_gru_e2e.py를 안 건드리고 상속해서 encode_flat_texts만
# 오버라이드한다. forward()가 self.encode_flat_texts(...)를 호출하는
# 구조라 이것만 바꾸면 나머지(GRU, step feature 등)는 그대로 재사용됨.
# ------------------------------------------------------------

class HierE2EModelV2(HierE2EModel):
    def __init__(self, *args, pooling="mean", **kwargs):
        super().__init__(*args, **kwargs)
        self.pooling = pooling

    def encode_flat_texts(self, flat_texts, device):
        enc = self.tokenizer(
            flat_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(device)
        out = self.encoder(**enc)
        if self.pooling == "cls":
            return out.last_hidden_state[:, 0, :]
        mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
        return (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


# ------------------------------------------------------------
# 레버 1: 랜덤 truncation Dataset (06번 CallTextDataset과 반환
# 시그니처가 같아서 collate_text를 그대로 재사용할 수 있다)
# ------------------------------------------------------------

class TruncatedCallTextDataset(Dataset):
    def __init__(self, jsonl_path, trunc_prob=0.5, trunc_min_len=3):
        self.calls = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.calls.append(json.loads(line))
        self.trunc_prob = trunc_prob
        self.trunc_min_len = trunc_min_len

    def __len__(self):
        return len(self.calls)

    def __getitem__(self, idx):
        call = self.calls[idx]
        texts = [clean_for_model(u["text"]) for u in call["utterances"]]
        speakers = [map_speaker(u["speaker"]) for u in call["utterances"]]
        n = len(texts)

        if n > self.trunc_min_len and random.random() < self.trunc_prob:
            cut = random.randint(self.trunc_min_len, n)
            texts, speakers, n = texts[:cut], speakers[:cut], cut

        return texts, speakers, call["label_vector"], n


# ------------------------------------------------------------
# 레버 2: 안정성 정칙화가 추가된 loss
# ------------------------------------------------------------

def compute_loss(logits, labels, lengths, device, pos_weight=None,
                  stability_weight=0.0, stability_margin=0.02):
    B, L, C = logits.shape
    mask = (torch.arange(L, device=device)[None, :] < lengths[:, None].to(device))

    labels_dev = labels.to(device)
    targets = labels_dev.unsqueeze(1).expand(B, L, C)
    pw = pos_weight.to(device) if pos_weight is not None else None
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
    bce = bce.mean(dim=2) * mask.float()
    per_call_bce = bce.sum(dim=1) / lengths.to(device).float()
    loss = per_call_bce.mean()

    stability_loss = torch.tensor(0.0, device=device)
    if stability_weight > 0 and L > 1:
        probs = torch.sigmoid(logits.float())
        drop = probs[:, :-1, :] - probs[:, 1:, :]           # (B, L-1, C): 직전 - 현재. 양수면 확률 하락
        drop = F.relu(drop - stability_margin)
        true_pos_mask = labels_dev.unsqueeze(1).expand(B, L - 1, C)   # 정답 클래스에만 적용
        step_mask = mask[:, 1:].unsqueeze(-1).float()                 # 유효 시점(패딩 제외)만
        penalty = (drop * true_pos_mask * step_mask).sum(dim=(1, 2))
        denom = (true_pos_mask * step_mask).sum(dim=(1, 2)).clamp(min=1.0)
        stability_loss = (penalty / denom).mean()
        loss = loss + stability_weight * stability_loss

    return loss, stability_loss.detach()


# ------------------------------------------------------------
# 학습 루프 (06_train_gru_e2e.py와 거의 동일한 구조)
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--backbone", default="klue/roberta-large")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--speaker_dim", type=int, default=16)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--pooling", choices=["mean", "cls"], default="mean",
                     help="발화 임베딩 풀링 방식 (mean=06번 기본 masked-mean, cls=CLS 토큰)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr_encoder", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--pos_weight_cap", type=float, default=5.0)
    ap.add_argument("--resume_from", default=None,
                     help="(선택) 이 체크포인트로 encoder/GRU 초기화 후 이어서 학습. "
                          "4-1 체크포인트에서 시작하고 싶으면 e2e_large/best.pt 지정.")
    ap.add_argument("--trunc_prob", type=float, default=0.5,
                     help="학습 샘플을 랜덤 길이로 자를 확률 (레버 1)")
    ap.add_argument("--trunc_min_len", type=int, default=3)
    ap.add_argument("--stability_weight", type=float, default=0.1,
                     help="안정성 정칙화 가중치 (레버 2, 0이면 끔)")
    ap.add_argument("--stability_margin", type=float, default=0.02)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_fp16", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[info] device={device}")
    print(f"[info] trunc_prob={args.trunc_prob}  stability_weight={args.stability_weight}")

    train_ds = TruncatedCallTextDataset(args.train_jsonl, args.trunc_prob, args.trunc_min_len)
    val_ds = CallTextDataset(args.val_jsonl)  # val은 항상 전체 길이로 평가 (06번과 동일 기준)
    print(f"[info] train {len(train_ds)}통화 / val {len(val_ds)}통화")

    pos_weight, pos_counts, n_train = compute_pos_weight(args.train_jsonl, cap=args.pos_weight_cap)
    for c, cnt, w in zip(SYMPTOM_CLASSES, pos_counts, pos_weight.tolist()):
        print(f"    {c:6s} 양성 {cnt:5d}/{n_train}  weight={w:.2f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_text)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_text)

    model = HierE2EModelV2(
        backbone=args.backbone, speaker_dim=args.speaker_dim, hidden=args.hidden,
        num_layers=args.num_layers, dropout=args.dropout, max_length=args.max_length,
        pooling=args.pooling,
    ).to(device)
    print(f"[info] pooling={args.pooling}")

    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(resume_ckpt["model_state"])
        print(f"[info] {args.resume_from} 에서 가중치 복원 완료")

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr_encoder},
        {"params": list(model.speaker_emb.parameters()) + list(model.gru.parameters())
                    + list(model.classifier.parameters()), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

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
        return cosine_after(step, 0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[encoder_lambda, head_lambda])
    print(f"[info] encoder만 웜업 {warmup_steps}/{total_steps} 스텝")

    use_fp16 = (device.type == "cuda") and not args.no_fp16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    print(f"[info] fp16 autocast = {use_fp16}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_stab, n_batches = 0.0, 0.0, 0

        for flat_texts, batch_idx, pos_idx, speaker_padded, labels, lengths, max_len in train_loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                logits = model(flat_texts, batch_idx, pos_idx, speaker_padded, lengths, max_len, device)
                loss, stab = compute_loss(
                    logits, labels, lengths, device, pos_weight=pos_weight,
                    stability_weight=args.stability_weight, stability_margin=args.stability_margin,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_stab += stab.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        train_elapsed = time.time() - t0

        model.eval()
        val_loss_total, val_batches = 0.0, 0
        with torch.no_grad():
            for flat_texts, batch_idx, pos_idx, speaker_padded, labels, lengths, max_len in val_loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                    logits = model(flat_texts, batch_idx, pos_idx, speaker_padded, lengths, max_len, device)
                    val_loss, _ = compute_loss(logits, labels, lengths, device, pos_weight=pos_weight)
                    val_loss_total += val_loss.item()
                val_batches += 1
        val_loss = val_loss_total / val_batches
        epoch_elapsed = time.time() - t0

        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} (안정성항 평균={total_stab / n_batches:.4f})  "
              f"val_loss={val_loss:.4f}  ({epoch_elapsed:.0f}s, train {train_elapsed:.0f}s)")

        ckpt = {
            "model_state": model.state_dict(),
            "config": {
                "backbone": args.backbone, "speaker_dim": args.speaker_dim,
                "hidden": args.hidden, "num_layers": args.num_layers,
                "dropout": args.dropout, "max_length": args.max_length,
                "symptom_classes": SYMPTOM_CLASSES, "e2e": True,
                "pooling": args.pooling,
            },
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> best.pt 갱신 (val_loss={val_loss:.4f})")

    print(f"\n[done] 저장 위치: {out_dir} (best.pt / last.pt) - 4-2 전용, 4-1 제출물과 무관")


if __name__ == "__main__":
    main()
