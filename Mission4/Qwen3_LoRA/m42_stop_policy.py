# -*- coding: utf-8 -*-
"""DCC Mission 4-2: streaming stopping policy for Qwen symptom probabilities.

No Qwen/torch/data dependence. This is a component to integrate into the
organizer's Predictor; it is NOT an inference.py submission on its own.

Only the present and prior nine probabilities influence stopping. Total call
length, labels and call IDs are never accessed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

SYMPTOMS = (
    '고열', '구토', '두통', '복통', '어지러움',
    '열상', '오심', '전신쇠약', '호흡곤란',
)


@dataclass(frozen=True)
class StopConfig:
    stable_count: int = 2
    confidence: float = 0.70
    max_wait: int = 28
    min_wait: int = 5
    symptom_threshold: float = 0.5  # Must stay fixed by contest rules.


class StreamingStopPolicy:
    """`observe` returns None (listen more) or a symptom list (stop).

    Even an empty symptom list represents a valid stopping decision.
    """

    def __init__(self, config: StopConfig = StopConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.previous_mask: Optional[int] = None
        self.same_count = 0
        self.latest: list[str] = []
        self.stopped = False

    def observe(self, probabilities) -> Optional[list[str]]:
        if self.stopped:
            raise RuntimeError('This call already stopped: reset() before a new call')
        p = [float(v) for v in probabilities]
        if len(p) != len(SYMPTOMS) or any(not math.isfinite(x) or x < 0 or x > 1 for x in p):
            raise ValueError('Expected exactly nine finite probabilities in [0,1]')

        threshold = self.config.symptom_threshold
        mask = sum((int(x >= threshold) << i) for i, x in enumerate(p))
        self.latest = [s for i, s in enumerate(SYMPTOMS) if (mask >> i) & 1]
        self.steps += 1
        self.same_count = self.same_count + 1 if mask == self.previous_mask else 1
        self.previous_mask = mask

        stable = (
            self.steps >= self.config.min_wait
            and mask != 0
            and self.same_count >= self.config.stable_count
            and max(p) >= self.config.confidence
        )
        if stable or self.steps >= self.config.max_wait:
            self.stopped = True
            return self.latest.copy()
        return None

    def finalize(self) -> list[str]:
        self.stopped = True
        return self.latest.copy()


if __name__ == '__main__':
    policy = StreamingStopPolicy()
    for i in range(5):
        p = [0.1] * 9
        if i >= 2:
            p[3] = 0.8  # 복통
        outcome = policy.observe(p)
        print(f'step={i+1}, stop={outcome is not None}, symptoms={outcome}')
        if outcome is not None:
            assert i + 1 == 5 and outcome == ['복통']
            break

    policy.reset()
    for i in range(28):
        result = policy.observe([0.1] * 9)
        if i < 27:
            assert result is None
        else:
            assert result == []
    assert policy.finalize() == []
    print('Policy self-test passed')
