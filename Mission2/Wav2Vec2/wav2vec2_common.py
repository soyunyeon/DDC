# =========================================================
# wav2vec2 학습(train) / 추론(inference) 공용 설정 + 함수
#
# 02_wav2vec2_prepare.py, 학습 노트북, inference.py가
# 모두 이 파일을 import해서 동일한 기준을 사용해야
# train/inference 간 전처리가 어긋나지 않는다.
#
# Mission1/Code/wav2vec2_common.py와 상수·chunk 분할 로직은
# 동일하다(같은 8kHz 한국어 전화통화 원본을 쓰므로 재조정 불필요).
# 다만 Mission2는 speaker 라벨이 이미 0/1 정수라
# gender 문자열<->정수 매핑은 필요 없어서 제외했다.
# =========================================================

# =========================================================
# 1. 모델 입력 기준값
# =========================================================

# 사전학습 wav2vec2 모델이 기대하는 sampling rate
TARGET_SR = 16000

# 이보다 짧은 청크는 이 길이까지 zero-padding
# (conv feature encoder가 최소 프레임을 뽑을 수 있는 길이 확보용)
MIN_DUR_SEC = 0.4

# 이보다 긴 세그먼트는 sliding window로 분할
# (배치 내 dynamic padding 비용 + self-attention 비용 억제 목적)
MAX_DUR_SEC = 8.0

# sliding window stride (겹치는 구간 4초 = 50% overlap)
STRIDE_SEC = 4.0


# =========================================================
# 2. 세그먼트 하나를 몇 개의 청크로 나눌지 계산
# =========================================================
#
# duration_sec <= MAX_DUR_SEC 인 경우
#   -> 자르지 않고 통째로 청크 1개
#      (짧은 경우 MIN_DUR_SEC padding은 로딩 시점에 처리)
#
# duration_sec > MAX_DUR_SEC 인 경우
#   -> MAX_DUR_SEC 길이 window를 STRIDE_SEC 간격으로 슬라이딩
#      마지막 window는 끝부분에 맞춰서 잘라 남는 구간이
#      버려지지 않게 한다.
# =========================================================

def compute_chunks(
    duration_sec
):

    if duration_sec <= MAX_DUR_SEC:

        return [
            (0.0, duration_sec)
        ]


    chunks = []

    start = 0.0

    while start + MAX_DUR_SEC < duration_sec:

        chunks.append(
            (
                start,
                start + MAX_DUR_SEC
            )
        )

        start += STRIDE_SEC


    # 마지막 window: 끝에 맞춰서 하나 더 추가
    last_start = duration_sec - MAX_DUR_SEC

    last_chunk = (
        last_start,
        duration_sec
    )

    if last_chunk not in chunks:

        chunks.append(
            last_chunk
        )


    return chunks
