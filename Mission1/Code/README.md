# Mission 1: 119 신고자 성별 분류

## 실행 방법

```bash
python inference.py --audio_dir {wav 폴더} --label_dir {json 폴더} --ckpt_path ./models/ecapa_ensemble_v3.ckpt --output ./outputs/mission1.csv
```

`--label_dir`의 JSON에서 startAt, endAt, speaker 정보만 사용합니다. gender 필드는 inference.py 어디에서도 읽지 않습니다.

## 폴더 구조

```
Mission1/
├── pretrained_models/
│   └── spkrec-ecapa-voxceleb/   (frozen ECAPA-TDNN, speechbrain 제공)
└── Code/
    ├── 01_extract_training.py
    ├── 02_common_preprocess.py
    ├── 03_extract_ecapa_embeddings.py
    ├── 04_final_train_ensemble.py
    ├── 05a_copy_provided_validation.py
    ├── 05b_preprocess_provided_validation.py
    ├── 05c_extract_embeddings_provided_validation.py
    ├── 05d_evaluate_provided_validation.py
    ├── inference.py                      ← 채점용 실행 스크립트
    ├── requirements.txt
    ├── Mission1_학습코드파일.ipynb
    ├── ecapa_ensemble_v3.ckpt
    └── README.md
```

01~03, 05a~05c는 원본 데이터에서 임베딩(shard)을 만드는 전처리 파이프라인이고, 04가 그 임베딩으로 최종 모델을 학습·저장합니다. 05d는 저장된 모델로 Data/Validation을 다시 예측해보는 선택적 검증 스크립트입니다. **채점에는 `inference.py` 한 번 실행으로 충분**하며, 나머지는 학습 과정을 재현하기 위한 코드입니다.

`pretrained_models` 폴더는 인터넷 연결 없이도 실행되도록 로컬에 포함했습니다. `inference.py`가 이 폴더를 먼저 찾고, 없을 때만 Hugging Face Hub에서 내려받습니다.

## 환경 설정

```bash
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

GPU가 없는 환경이면 `+cu121` 없이 일반 CPU 빌드를 설치해도 됩니다. `inference.py`가 `torch.cuda.is_available()`로 자동 감지해서 CPU로 전환합니다.

## 모델 구성 요약

frozen ECAPA-TDNN(speechbrain/spkrec-ecapa-voxceleb)으로 신고자 발화 segment마다 192차원 임베딩을 추출하고, 그 위에 MLP(은닉층 128개) 10개를 서로 다른 random seed로 학습시켜 평균 낸 앙상블로 성별을 예측합니다. segment별 예측 확률은 duration에 비례한 가중평균으로 통화 단위로 집계하고, val_calls(학습에 쓰이지 않은 held-out)로 학습한 **Platt scaling(확률 calibration)**을 적용한 뒤, **Decision Threshold 0.5**(대회 규정상 고정값 — Train/Validation 어느 쪽으로도 탐색·조정하지 않음)를 적용해 최종 M/F를 결정합니다. calibration은 대회 공지 "Decision Threshold 관련 기준" Q4에서 허용하는 범위로, 확률값 자체를 보정할 뿐 cutoff(0.5)는 그대로 유지합니다.

검증 결과: 팀 공통 held-out(val_calls, 5,840통화) **98.07%**, 대회 제공 Validation(3,640통화) **98.27%**.

## 추론 시간 관련 안내

**(1) 모델 파라미터 수**

- Frozen ECAPA-TDNN: `[[ECAPA_PARAM_COUNT]]`개 (아래 스니펫으로 직접 확인)
- 분류기(MLP 10개 앙상블): 24,833개 × 10 = 248,330개
- Probability calibration(Platt scaling, 1-feature 로지스틱 회귀): 2개 (weight 1 + bias 1, 무시 가능한 수준)
- Total = Active = 위 세 값의 합 (frozen 구간도 forward pass에는 전부 관여하므로 sparse/MoE 구조 아님)

파라미터 수 확인용 스니펫:

```python
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="../pretrained_models/spkrec-ecapa-voxceleb",
)
print(sum(p.numel() for p in classifier.mods.embedding_model.parameters()))
```

**(2) 학습 및 추론 환경**

- GPU: NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)
- Python 3.11, PyTorch 2.5.1 (CUDA 12.1)

**(3) Validation 추론 시 batch size**

고정 배치 크기를 쓰지 않고, 오디오 길이 기반 동적 배치를 사용했습니다. 배치 내 오디오 길이 합이 32초를 넘지 않는 선에서 최대 32개까지 묶고, 긴 오디오는 자동으로 배치 크기를 줄입니다. (고정 배치를 썼을 때 긴 오디오가 섞이면 VRAM이 4GB 한계에 가까워지며 처리 속도가 100배 이상 느려지는 문제를 겪어서, 이를 해결하는 과정에서 도입한 방식입니다.)

**(4) 측정된 추론 시간**

Data/Validation 3,640통화(58,102개 신고자 segment) 기준, RTX 3050 4GB에서 전체 추론 **391.5초(약 6.5분)**.

- 통화당 평균: 약 0.108초
- segment당 평균: 약 6.74ms

(calibration 단계는 통화 3,640개의 확률값에 1-feature 로지스틱 회귀를 적용하는 연산이라 전체 추론 시간에 미치는 영향은 무시할 수준이라 별도로 재측정하지 않았습니다.)
