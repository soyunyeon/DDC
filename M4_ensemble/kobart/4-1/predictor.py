import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from m4_model import load_checkpoint


OUTPUT_LABELS = [
    "고열", "구토", "두통", "복통", "어지러움",
    "열상", "오심", "전신쇠약", "호흡곤란"
]


class Predictor:
    """
    KoBART Mission 4-1 streaming adapter.

    step(input_text)는 현재 발화까지 누적된,
    학습 때와 동일하게 전처리된 input_text를 받는다.

    원시 발화만 넣는 인터페이스는 아니다.
    """

    def __init__(self, checkpoint=None, device=None):
        self.checkpoint = Path(
            checkpoint or ROOT / "checkpoint"
        ).resolve()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        with open(
            ROOT / "label_map.json",
            encoding="utf-8"
        ) as f:
            self.labels = json.load(f)["labels"]

        assert len(self.labels) == 9
        assert set(self.labels) == set(OUTPUT_LABELS)

        self.output_indices = [
            self.labels.index(name)
            for name in OUTPUT_LABELS
        ]

        with open(
            self.checkpoint / "metadata.json",
            encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        self.max_input = metadata.get("max_input")

        if not isinstance(self.max_input, int) or self.max_input <= 0:
            raise ValueError(
                "metadata.json의 max_input 확인 필요. "
                "기존 평가 코드와 동일하게 설정해야 합니다."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint)
        )
        self.tokenizer.truncation_side = "left"

        self.model = load_checkpoint(
            str(self.checkpoint),
            str(self.device)
        )
        self.model.eval()

        self.reset()

    def reset(self):
        self.last_probs = None
        self.step_count = 0

    @torch.inference_mode()
    def predict_batch(self, texts):
        if not texts:
            return []

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_input,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(
            self.device
        )

        probs = self.model.predict_probs(
            input_ids,
            attention_mask
        )

        probs = probs.detach().float().cpu()

        assert probs.ndim == 2
        assert probs.shape == (len(texts), 9)

        result = []

        for row in probs.tolist():
            ordered = [
                float(row[i]) for i in self.output_indices
            ]

            assert all(
                math.isfinite(x) and 0 <= x <= 1
                for x in ordered
            )

            result.append(ordered)

        return result

    def step(self, input_text):
        """
        input_text: 현재 발화까지의 누적 전처리 텍스트.
        반환: 요청한 클래스 순서의 확률 딕셔너리.
        """
        probs = self.predict_batch([input_text])[0]

        self.last_probs = dict(zip(OUTPUT_LABELS, probs))
        self.step_count += 1

        return self.last_probs.copy()

    def finalize(self):
        # 통화 종료 시 새로운 미래 정보를 이용하지 않음
        if self.last_probs is None:
            return None
        return self.last_probs.copy()
