# ============================================================
# Mission 4-1 개선 - soft Jaccard 손실 추가 (기존 e2e_large에서 resume)
#
# 혼동분석(15_confusion_analysis.py) 결과: 오심이 특정 클래스 하나와
# 짝으로 헷갈리는 게 아니라, 여러 클래스(복통/전신쇠약/어지러움 등)에
# 걸쳐 예측 집합이 정답보다 전체적으로 비대해지는 패턴이었다(precision
# 0.293 << recall 0.389). 클래스 독립적인 BCE는 이 "집합 크기" 자체를
# 벌주지 않으므로, 채점식(순간마다 |예측∩정답|/합집합과 동치인 클래스별
# 이진 F1 평균)을 손실에 직접 반영하는 soft Jaccard 항을 추가한다.
#
# 손실 = BCE + lambda_jaccard * (1 - soft_jaccard)
#   soft_jaccard = sum_c(p_c * y_c) / sum_c(p_c + y_c - p_c*y_c)
#   (p = sigmoid 출력, y = 0/1 정답. 전부 미분 가능 - 0.5 threshold는
#    여전히 추론/판정에만 쓰이고 손실 계산에는 전혀 등장하지 않으므로
#    "Decision Threshold 0.5 고정" 규정과 무관하다.)
#
# gold가 빈 심볼텀인 콜은 01_common_preprocess.py 단계에서 이미 전부
# 제외되므로(label_vector 합이 항상 >=1), union이 0이 되는 경우는
# 없다 - 별도 valid-mask 없이 clamp만으로 안전하다.
#
# 06_train_gru_e2e.py와 동일 아키텍처 골격을 그대로 재사용(상속 없이
# 모듈 재사용 - 14번 스크립트와 같은 패턴). --resume_from으로 이미
# 수렴한 e2e_large/best.pt에서 시작하는 걸 전제로 하므로, 기본값을
# 짧은 추가학습(1 epoch, 낮은 lr, warmup 없음)에 맞춰뒀다 - 처음부터
# 학습할 때 쓰는 하이퍼파라미터(06번 기본값)를 그대로 쓰면 이미 수렴한
# encoder가 다시 크게 흔들린다.
#
# 사용법 (lambda_jaccard는 팀 val에서 스윕 - 1.0, 2.0, 4.0 추천)
#   python 16_train_gru_e2e_jaccard.py \
#       --train_jsonl ../Work/Common/mission4_train.jsonl \
#       --val_jsonl ../Work/Common/mission4_val.jsonl \
#       --resume_from ../Work/ckpt/e2e_large/best.pt \
#       --output_dir ../Work/ckpt/e2e_large_sj2 \
#       --lambda_jaccard 2.0
# ============================================================

import argparse
import importlib.util
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("train_gru_e2e_module", _THIS_DIR / "06_train_gru_e2e.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

HierE2EModel = _m.HierE2EModel
CallTextDataset = _m.CallTextDataset
collate_text = _m.collate_text
compute_pos_weight = _m.compute_pos_weight
SYMPTOM_CLASSES = _m.SYMPTOM_CLASSES


def compute_loss(logits, labels, lengths, device, pos_weight=None, lambda_jaccard=0.0):
    B, L, C = logits.shape
    mask = (torch.arange(L, device=device)[None, :] < lengths[:, None].to(device))
    mask_f = mask.float()

    labels_dev = labels.to(device)
    targets = labels_dev.unsqueeze(1).expand(B, L, C)
    pw = pos_weight.to(device) if pos_weight is not None else None
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
    bce = bce.mean(dim=2) * mask_f
    bce_loss = (bce.sum(dim=1) / lengths.to(device).float()).mean()

    jaccard_loss = torch.tensor(0.0, device=device)
    if lambda_jaccard > 0:
        probs = torch.sigmoid(logits.float())        # fp16 autocast 하에서도 안정적인 계산을 위해 fp32로
        y = targets.float()
        intersection = (probs * y).sum(dim=2)          # (B, L)
        union = probs.sum(dim=2) + y.sum(dim=2) - intersection
        soft_jaccard = intersection / union.clamp(min=1e-6)
        per_step_loss = (1.0 - soft_jaccard) * mask_f
        jaccard_loss = (per_step_loss.sum(dim=1) / lengths.to(device).float()).mean()

    loss = bce_loss + lambda_jaccard * jaccard_loss
    return loss, bce_loss.detach(), jaccard_loss.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--resume_from", required=True,
                     help="이어서 학습할 시작 체크포인트. 보통 ckpt/e2e_large/best.pt")
    ap.add_argument("--lambda_jaccard", type=float, required=True,
                     help="soft Jaccard 항 가중치. 팀 val에서 1.0/2.0/4.0 스윕 추천")
    ap.add_argument("--backbone", default="klue/roberta-large")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--speaker_dim", type=int, default=16)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1,
                     help="이미 수렴한 체크포인트에서 짧게 이어 학습 (팀원 실험 기준 1epoch로 충분)")
    ap.add_argument("--lr_encoder", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.0,
                     help="resume 전제라 기본 0 - encoder가 이미 적응돼 있어 재웜업 불필요")
    ap.add_argument("--pos_weight_cap", type=float, default=5.0,
                     help="e2e_large와 동일하게 유지 - 자카드 손실 효과를 pos_weight 변화와 섞지 않기 위해 건드리지 않음")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_fp16", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[info] device={device}  lambda_jaccard={args.lambda_jaccard}")

    train_ds = CallTextDataset(args.train_jsonl)
    val_ds = CallTextDataset(args.val_jsonl)
    print(f"[info] train {len(train_ds)}통화 / val {len(val_ds)}통화")

    pos_weight, pos_counts, n_train = compute_pos_weight(args.train_jsonl, cap=args.pos_weight_cap)
    for c, cnt, w in zip(SYMPTOM_CLASSES, pos_counts, pos_weight.tolist()):
        print(f"    {c:6s} 양성 {cnt:5d}/{n_train}  weight={w:.2f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_text)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_text)

    model = HierE2EModel(
        backbone=args.backbone, speaker_dim=args.speaker_dim, hidden=args.hidden,
        num_layers=args.num_layers, dropout=args.dropout, max_length=args.max_length,
    ).to(device)

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
        return 0.5 * (1 + __import__("math").cos(__import__("math").pi * min(max(progress, 0.0), 1.0)))

    def encoder_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return cosine_after(step, warmup_steps)

    def head_lambda(step):
        return cosine_after(step, 0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[encoder_lambda, head_lambda])

    use_fp16 = (device.type == "cuda") and not args.no_fp16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    print(f"[info] fp16 autocast = {use_fp16}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_bce, total_jacc, n_batches = 0.0, 0.0, 0.0, 0

        for flat_texts, batch_idx, pos_idx, speaker_padded, labels, lengths, max_len in train_loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                logits = model(flat_texts, batch_idx, pos_idx, speaker_padded, lengths, max_len, device)
                loss, bce_l, jacc_l = compute_loss(
                    logits, labels, lengths, device,
                    pos_weight=pos_weight, lambda_jaccard=args.lambda_jaccard,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_bce += bce_l.item()
            total_jacc += jacc_l.item()
            n_batches += 1

        train_elapsed = time.time() - t0

        model.eval()
        val_loss_total, val_bce_total, val_jacc_total, val_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for flat_texts, batch_idx, pos_idx, speaker_padded, labels, lengths, max_len in val_loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                    logits = model(flat_texts, batch_idx, pos_idx, speaker_padded, lengths, max_len, device)
                    vloss, vbce, vjacc = compute_loss(
                        logits, labels, lengths, device,
                        pos_weight=pos_weight, lambda_jaccard=args.lambda_jaccard,
                    )
                    val_loss_total += vloss.item()
                    val_bce_total += vbce.item()
                    val_jacc_total += vjacc.item()
                val_batches += 1
        val_loss = val_loss_total / val_batches
        epoch_elapsed = time.time() - t0

        print(f"[epoch {epoch}/{args.epochs}] "
              f"train_loss={total_loss / n_batches:.4f} (bce={total_bce / n_batches:.4f}, jacc={total_jacc / n_batches:.4f})  "
              f"val_loss={val_loss:.4f} (bce={val_bce_total / val_batches:.4f}, jacc={val_jacc_total / val_batches:.4f})  "
              f"({epoch_elapsed:.0f}s, train {train_elapsed:.0f}s)")

        ckpt = {
            "model_state": model.state_dict(),
            "config": {
                "backbone": args.backbone, "speaker_dim": args.speaker_dim,
                "hidden": args.hidden, "num_layers": args.num_layers,
                "dropout": args.dropout, "max_length": args.max_length,
                "symptom_classes": SYMPTOM_CLASSES, "e2e": True,
            },
            "lambda_jaccard": args.lambda_jaccard,
            "resume_from": args.resume_from,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> best.pt 갱신 (val_loss={val_loss:.4f})")

    print(f"\n[done] 저장 위치: {out_dir} (best.pt / last.pt)  lambda_jaccard={args.lambda_jaccard}")
    print("[info] 실제 4-1 점수는 07_eval_fast.py로 CSV를 뽑은 뒤 score.py --mission 4-1 로 확인할 것 "
          "(val_loss는 BCE+jaccard 혼합값이라 lambda가 다른 run끼리 직접 비교 불가)")


if __name__ == "__main__":
    main()
