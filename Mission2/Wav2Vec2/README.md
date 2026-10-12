# Mission 2 - Wav2Vec2 Speaker Classification Share

## Best Model

- Backbone: Kkonjeong/wav2vec2-base-korean (facebook/wav2vec2-base, Zeroth-Korean 한국어 적응, 9,400만 파라미터)
- Freeze 전략: feature encoder(CNN) 항상 freeze, transformer encoder는 마지막 4개 layer만 학습(N_UNFROZEN_LAYERS=4), classifier head(projector+classifier)는 항상 학습
- 내부 val split 기준 best val_segment_level_acc: 0.9189536136273574 (`04_train_wav2vec2_colab.ipynb` 로그 및 체크포인트 내부 기록 일치 확인됨)
- 공식 Validation(3,640통화) segment-level accuracy: 0.9175 (`06_score_official_validation.py`로 재채점 확인)

**주의**: 중간에 컨텍스트 마진(±0.3초, "실험2")을 적용한 버전이 있었으나, annotation은 startAt/endAt만 허용한다는 대회 규정에 위반되어 **폐기**됨. 이 폴더의 모델/코드는 전부 마진 없는(실험1 파이프라인) 최종 버전이다.

## Input

`startAt`, `endAt`만 사용해서 원본 wav에서 발화 조각(segment)을 crop한다. `speaker`는 정답 라벨이므로 모델 입력으로 절대 사용하지 않는다.

## Label

- 0 = 119대원
- 1 = 신고자

## Model Configuration

- pretrained_ckpt: `Kkonjeong/wav2vec2-base-korean`
- num_labels: 2
- N_UNFROZEN_LAYERS: 4
- 컨텍스트 마진: 없음 (startAt/endAt 그대로 crop)
- sampling rate: 16000 (8kHz 원본은 on-the-fly 리샘플)
- 세그먼트 최소 길이: 0.4초 미만은 zero-padding
- 세그먼트 최대 길이: 8초 초과 시 4초 stride(50% overlap) sliding window로 분할, 평가/추론 시 청크별 확률(또는 pooled feature)을 평균해서 segment 단위로 집계
- batch size: 학습 16 / 추론·export 32
- optimizer: AdamW (backbone lr=2e-5, head lr=1e-3), linear warmup+decay scheduler
- loss: CrossEntropyLoss (train split 기준 class weight 적용)
- epochs: 4

## Internal Split (train/val, chunk 기준)

- Train: 717,537
- Val: 180,989

## Files

- `mission2_wav2vec2_best.pt` (git-lfs): 최종 체크포인트. `{model_state_dict, pretrained_ckpt, num_labels, val_segment_level_acc}` 딕셔너리로 저장됨.
- `model.py`: 모델 정의 + freeze 전략 (`build_model()`, 학습용)
- `wav2vec2_common.py`: TARGET_SR/MIN_DUR_SEC/MAX_DUR_SEC/STRIDE_SEC 상수 + `compute_chunks()` — 학습/추론/export 전부 공용
- `inference.py`: 대회 제출용 추론 스크립트 (`python inference.py --audio_dir --label_dir --ckpt_path --output`)
- `04_train_wav2vec2_colab.ipynb`: 최종 체크포인트를 만든 학습 노트북 (Colab, 실행 로그 포함 — epoch별 train_loss/val_loss/val_segment_level_acc 그대로 보존됨)
- `07_export_fusion_predictions.ipynb`: Fusion(Audio+Text) 팀원용 — segment 단위 `audio_prob`(speaker=1일 확률) + pooling 후 256-d 임베딩을 csv/npy로 export하는 Colab 노트북
- `08_export_frame_features.py`: Cross-Attention 실험용 — pooling 전 frame/sequence-level feature(`--feature_level projector`=256-d 또는 `encoder`=768-d)를 segment/chunk별 .npy로 export하는 독립 실행 스크립트 (Colab/Drive 의존성 없음, `--ckpt_path`로 체크포인트 지정)
- `pretrained/Kkonjeong-wav2vec2-base-korean/`: `config.json` + `preprocessor_config.json` (오프라인 환경에서 `AutoConfig`/`Wav2Vec2FeatureExtractor`가 HF Hub 접근 없이 로드할 수 있도록 로컬 사본 제공 — 단, 현재 `inference.py`는 아직 이 로컬 경로가 아니라 Hub ID로 직접 호출하도록 돼 있어 완전 오프라인 실행을 보장하려면 이 부분 추가 수정 필요)
- `fusion_export/`: `07_export_fusion_predictions.ipynb` 최종 실행 결과
  - `audio_{train,val}_predictions.csv`: call_id, segment_id, speaker, audio_prob(=speaker=1일 확률)
  - `audio_{train,val}_embeddings.npy` (+`_index.csv`로 행 순서 명시): pooling 후 256-d 임베딩, float16
  - train/val은 팀 공통 fixed split(`Shared/train_calls.csv`/`val_calls.csv`) 기준

## Reproduce

```bash
pip install -r requirements.txt
```

학습 재현: `04_train_wav2vec2_colab.ipynb`를 Colab GPU 런타임에서 실행 (Drive에 shard/manifest 업로드 필요).

추론: `python inference.py --audio_dir <wav_dir> --label_dir <json_dir> --ckpt_path mission2_wav2vec2_best.pt --output ./outputs/mission2.csv`

Frame-level feature 추출(Cross-Attention용): `python 08_export_frame_features.py --ckpt_path mission2_wav2vec2_best.pt --output_dir <out_dir> --feature_level encoder`
