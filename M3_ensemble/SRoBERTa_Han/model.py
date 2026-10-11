# ============================================================
# Mission 3 - SRoBERTa 모델 정의 (발화 단위 pooling 분류기)
#
# 구조
# 1. 콜을 발화 단위로 분해해서 각 발화를 encoder에 통과
# 2. token-level masked mean pooling -> 발화 임베딩
# 3. 발화 임베딩들을 콜 단위로 masked mean + max pooling -> 콜 표현
# 4. Linear head -> 9개 증상 logit
#
# encoder 아키텍처는 assets/config.json에서 읽어서 만들고,
# 가중치는 체크포인트(state_dict)에서 전부 불러온다.
# 따라서 채점 환경에서 인터넷이나 HuggingFace 캐시가 필요 없다.
# (체크포인트에 encoder 전체 가중치가 들어 있음)
# ============================================================

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class HierarchicalClassifier(nn.Module):

    def __init__(self, assets_dir, num_labels=9):

        super().__init__()

        # 로컬 config로 encoder 구조만 생성 (사전학습 가중치 다운로드 없음)
        config = AutoConfig.from_pretrained(str(assets_dir), local_files_only=True)
        self.encoder = AutoModel.from_config(config)

        hidden_size = config.hidden_size

        # mean pooling + max pooling을 이어붙이므로 입력 차원은 hidden_size * 2
        self.head = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, utt_mask):
        """
        input_ids      : (B, U, L)  콜 B개, 콜당 발화 U개, 발화당 토큰 L개
        attention_mask : (B, U, L)
        utt_mask       : (B, U)     실제 발화 1, 패딩 발화 0
        """

        B, U, L = input_ids.shape

        flat_ids = input_ids.view(B * U, L)
        flat_mask = attention_mask.view(B * U, L)

        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        last_hidden = outputs.last_hidden_state  # (B*U, L, H)

        # token-level masked mean pooling -> 발화 임베딩
        tok_mask = flat_mask.unsqueeze(-1).float()
        summed = (last_hidden * tok_mask).sum(1)
        counts = tok_mask.sum(1).clamp(min=1e-9)
        utt_emb = summed / counts  # (B*U, H)

        H = utt_emb.shape[-1]
        utt_emb = utt_emb.view(B, U, H)

        um = utt_mask.unsqueeze(-1)  # (B, U, 1)

        # 콜 단위 masked mean pooling
        summed_u = (utt_emb * um).sum(1)
        counts_u = um.sum(1).clamp(min=1e-9)
        mean_call = summed_u / counts_u

        # 콜 단위 masked max pooling (패딩 발화는 아주 작은 값으로 채워서 제외)
        neg_fill = torch.finfo(utt_emb.dtype).min
        utt_emb_masked = utt_emb.masked_fill(um == 0, neg_fill)
        max_call, _ = utt_emb_masked.max(dim=1)
        max_call = torch.where(
            torch.isinf(max_call), torch.zeros_like(max_call), max_call
        )

        call_repr = torch.cat([mean_call, max_call], dim=-1)

        return self.head(call_repr)
