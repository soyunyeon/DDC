# ============================================================
# 발화 텍스트 노이즈 정리 - 모델에 넣기 직전에 적용
#
# [개인정보] 등 비식별화 placeholder는 전체 발화의 11.7%를 차지하지만
# 증상 정보가 전혀 없는 순수 노이즈라, 모델 입력 직전에 제거한다.
#
# "샘플마다 다른 전처리 금지" 규정과 충돌하지 않는다 - 특정 라벨/샘플을
# 보고 분기하는 게 아니라, 모든 샘플에 동일한 규칙(이 placeholder
# 문자열은 지운다)을 적용하는 일반 규칙이기 때문이다.
#
# 공통 전처리(01_common_preprocess.py)의 산출물(mission4_*.jsonl)은
# 건드리지 않는다 - 팀 공용 산출물이라 임의로 바꾸지 않고, 이 모듈을
# 쓰는 쪽(임베딩 추출, 학습, 추론)에서 텍스트를 쓰기 직전에만 적용한다.
# 학습 때 썼으면 추론(Predictor.step) 때도 반드시 똑같이 적용해야
# 한다 - 안 그러면 학습-추론 입력 분포가 어긋난다.
# ============================================================

import re

_NOISE_PATTERNS = [
    re.compile(r"\[개인정보\]"),
]


def clean_for_model(text: str) -> str:
    if not text:
        return text or ""
    cleaned = text
    for pat in _NOISE_PATTERNS:
        cleaned = pat.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
