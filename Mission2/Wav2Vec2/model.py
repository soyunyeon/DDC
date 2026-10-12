import torch
from transformers import Wav2Vec2ForSequenceClassification


# =========================================================
# Mission 2 - 신고자(speaker=1) / 119대원(speaker=0) 화자 분류 모델
# =========================================================
#
# 사전학습 체크포인트: Kkonjeong/wav2vec2-base-korean
#   - facebook/wav2vec2-base(base 아키텍처, 9,400만 파라미터)를
#     Zeroth-Korean 데이터로 한국어에 적응시킨 모델
#   - Mission1에서 kresnik/wav2vec2-large-xlsr-korean(large, 3.15억
#     파라미터)는 1 epoch에 12시간 이상 걸리는 걸 확인하고 base로
#     교체해 속도를 확보한 바 있어, 동일 체크포인트를 재사용함
#
# 분류 head: HuggingFace Wav2Vec2ForSequenceClassification의
#   기본 head(hidden state pooling + linear classifier)를 그대로 사용.
#   pretrained 체크포인트에 없는 projector/classifier 레이어는
#   무작위 초기화된 뒤 이번 학습으로 새로 학습됨.
#
# Freeze 전략 (Mission1과 동일):
#   - feature encoder(CNN, 파형 -> 프레임 특징 추출부)는 항상 freeze
#     (사전학습된 저수준 음향 특징 그대로 사용)
#   - transformer encoder는 마지막 N_UNFROZEN_LAYERS개 layer만 학습,
#     나머지는 freeze (데이터 규모 대비 과적합 방지 + 학습 시간 단축)
#   - 분류 head(projector, classifier)는 항상 학습
#
# Mission1과의 차이는 라벨 정의뿐 (gender M/F -> speaker 0/1).
# speaker는 라벨링데이터에 이미 0/1 정수로 들어있어 별도 인코딩
# 매핑(GENDER_TO_LABEL 같은 것)이 필요 없음.
# =========================================================

PRETRAINED_CKPT = "Kkonjeong/wav2vec2-base-korean"
NUM_LABELS = 2

# 실험2의 컨텍스트 마진(±0.3초)이 대회 규정 위반으로 확인되어 폐기.
# 실험2에서 같이 바뀌었던 N_UNFROZEN_LAYERS(2->4)만 분리해서
# 실험1의 데이터 파이프라인(마진 없는 세그먼트 그대로) 위에 재실험.
N_UNFROZEN_LAYERS = 4


def build_model(
    pretrained_ckpt=PRETRAINED_CKPT,
    num_labels=NUM_LABELS,
    n_unfrozen_layers=N_UNFROZEN_LAYERS
):
    """
    학습용 모델을 만든다 (freeze 전략 적용, 학습 대상 파라미터만
    requires_grad=True로 설정된 상태로 반환).

    추론(inference.py)에서는 이 함수 대신, 저장된 체크포인트의
    config로 빈 모델을 만들고 state_dict를 그대로 불러오는 방식을
    쓴다 (freeze 여부는 추론에 영향 없으므로 불필요).
    """

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        pretrained_ckpt,
        num_labels=num_labels
    )

    # CNN feature encoder는 항상 freeze
    model.freeze_feature_encoder()

    # transformer encoder: 마지막 n_unfrozen_layers개만 학습
    total_layers = len(model.wav2vec2.encoder.layers)

    for i, layer in enumerate(model.wav2vec2.encoder.layers):

        requires_grad = (
            i >= total_layers - n_unfrozen_layers
        )

        for p in layer.parameters():
            p.requires_grad = requires_grad

    return model


if __name__ == "__main__":
    # 간단한 동작 확인용
    model = build_model()

    n_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    n_total = sum(p.numel() for p in model.parameters())

    print(f"학습 대상 파라미터: {n_trainable:,} / 전체 {n_total:,}")
