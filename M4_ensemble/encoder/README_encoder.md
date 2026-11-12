# Encoder (KLUE-RoBERTa) 앙상블용 자료

요청 항목 순서대로 정리했다. 점수는 전부 **공식 score.py 함수로 계산한 값**(proxy 아님)이고, 판정은 0.5 고정, 0.5 이상이 없으면 빈 리스트다.

## Mission 4-1

### 1. val_holdout probs.csv

| 파일 | 모델 | 용도 |
|:--|:--|:--|
| `m41_val_holdout_probs_klue_sj1.csv` | KLUE-RoBERTa, BCE + soft Jaccard | **4-1 제출 모델** |
| `m41_val_holdout_probs_klue_base.csv` | KLUE-RoBERTa, BCE | 4-2 제출 모델 (4-1 앙상블 후보로도 사용 가능) |
| `official_probs_klue_sj1.csv`, `official_probs_klue_base.csv` | 위와 같음 | 공식 Validation 3,640건 (평가 확인용) |

- 컬럼: `label_file,step,고열,구토,두통,복통,어지러움,열상,오심,전신쇠약,호흡곤란` (요청 형식 그대로)
- val_holdout = 팀 고정 split의 val 5,840건 (`val_calls.csv`). 공통 전처리 결과 그대로 썼다.
- step = 1부터 시작하는 누적 발화 수. t행 확률은 발화 1..t만 보고 낸 값이다.
- 발화 순서는 공식 inference.py, score.py와 같은 (startAt, endAt) 정렬이다. 공통 전처리도 같은 정렬이라 label_file + step으로 그대로 합쳐진다.
- 제출 inference.py로 공식 Validation 300건을 직접 돌려 이 확률과 비교했다: 판정 99.81% 일치 (남은 차이는 bf16 묶음 계산과 한 건씩 계산의 소수점 오차).

| 4-1 점수 | val_holdout | 공식 Validation |
|:--|:--|:--|
| klue_sj1 (제출) | 0.4798 | 0.4817 |
| klue_base | 0.4515 | 0.4490 |

### 2. 최종 제출용

| 항목 | 내용 |
|:--|:--|
| 체크포인트 | `ckpt/klue_encoder.pt` (846MB). 4-1 모델과 4-2 모델이 둘 다 들어 있다. **GitHub 100MB 제한을 넘어서 따로 전달** |
| 모델 코드 | `encoder_model.py` (`StreamingEncoder` 클래스: 체크포인트 로드, 입력 만들기, 스트리밍 추론) |
| Predictor 구현 | `predictor_block.py` (공식 inference.py의 Predictor 클래스 전체: load_model, reset, step, finalize) |
| 클래스 순서 | 공식 `SYMPTOM_CLASSES` 순서와 **같다**. 체크포인트에도 클래스 순서가 저장돼 있고, 로드할 때 다르면 에러를 낸다. 별도 label_map 없음 |
| 전처리 | 따로 없음. 오디오, 임베딩 추출 안 씀. 텍스트 정리(공백 정리)와 화자 표시는 `encoder_model.py` 안에서 한다 |
| 패키지 | `requirements.txt` (Python 3.11, torch 2.6.0, transformers 5.18.0) |

**스트리밍 처리 방식 (재현에 필요한 핵심)**
- 히든 스테이트를 누적하지 않는다. 발화마다 토큰 조각을 버퍼에 쌓고, 매 step마다 1..t번째 발화를 이어 붙인 입력 전체를 새로 인코딩한다.
- 발화 하나 = `" [대원] 텍스트"` 또는 `" [신고자] 텍스트"` (speaker 0 = 대원, 1 = 신고자)를 따로 토큰화.
- 이어 붙인 길이가 512토큰(특수 토큰 포함)을 넘으면 앞 128토큰 + 가장 최근 부분만 남기고 `[CLS] ... [SEP]`로 감싼다.
- 출력은 sigmoid 확률 9개. 발화당 약 14ms (RTX 3050 4GB, bf16). CPU에서는 훨씬 느리다.
- 쓰는 정보는 `utt.text`, `utt.speaker`뿐이다.

**팀 inference.py에 넣는 법 (예시)**

```python
# load_model 안에서 (encoder_model.py가 inference.py와 같은 폴더에 있다고 가정)
from encoder_model import StreamingEncoder
enc_path = os.path.join(os.path.dirname(ckpt_path), "klue_encoder.pt")
self.enc = StreamingEncoder.load(enc_path, device=self.device, mission=self.mission)
#   mission="4-1" -> soft Jaccard 모델, mission="4-2" -> BCE 모델

# reset 안에서
self.enc.reset()

# step 안에서
p_enc = self.enc.add(utt.speaker, utt.text)   # numpy (9,), SYMPTOM_CLASSES 순서, 발화 1..t 기준
```

### 3. 하이퍼파라미터 / config
- 모델 구조 설정(config.json)과 토크나이저는 체크포인트 안에 들어 있어서 따로 줄 파일이 없다. 인터넷 없이 로드된다.
- 입력 설정(max_len 512, head 128)과 4-2 중단 정책(t_min 6, k 3)도 체크포인트에 저장돼 있다. 같은 내용을 `meta.json`에 적어 두었다.

## Mission 4-2

### 1. 중단 정책
- 코드: `predictor_block.py`의 `step()` 4-2 부분 (제출용), `sim_42.py` (저장된 확률로 정책을 고르는 시뮬레이터)
- 규칙: 지금까지 받은 발화 수 t ≥ **t_min 6**이고, 확률 0.5 이상인 증상 집합이 비어 있지 않으면서 **k = 3번 연속 같으면** 중단하고 그 집합을 반환
- 끝까지 중단하지 않으면 마지막 시점의 0.5 이상 증상 집합 (비어 있을 수 있음)
- 중단 판단 기준과 판정 기준 모두 0.5. 별도 confidence 점수는 쓰지 않는다. 전체 길이는 쓰지 않는다
- t_min ∈ {5, 6, 7, 8, 10, 12, 15, 20}, k ∈ {1, 2, 3, 4}를 val_holdout에서 시뮬레이션해서 골랐다
- 확률은 BCE 모델(`klue_base`) 것을 쓴다. soft Jaccard 모델은 일찍 확신해서 덜 정확한 시점에 멈추기 때문에 4-2에서 더 낮았다 (val_holdout 0.4258)

### 2. val_holdout 통화별 4-2 최종 예측
- `m42_val_holdout_pred_klue_base.csv`: 통화당 1행, 5,840행
- 컬럼: `label_file, halt_step, symptom`(공식 형식, `;`로 연결), 9개 증상 0/1, `n_utt_scoring_only`(통화 전체 발화 수, 채점과 분석용이며 정책에는 쓰지 않음)

### 3. 현재 4-2 성능 (공식 채점 함수 기준)

| | val_holdout | 공식 Validation |
|:--|:--|:--|
| 4-2 점수 | 0.4422 | 0.4464 |
| Macro-F1 | 0.5512 | 0.5576 |
| 평균 조기판단력 | 0.8021 | 0.8006 |

- 평가 방식: 공식 score.py의 `macro_f1`, `earliness_weight`를 그대로 import해서 계산 (score.py 4-2 채점과 값이 같은지 확인함)
- 공식 Validation은 정책을 고른 뒤 평가만 했다
- 오심 F1이 0.12로 가장 낮다. 원인은 라벨 불일치 (오심 통화의 85%는 대화에 메스꺼움 표현이 없음). 자세한 분석은 `M4_Encoder_실험결과.md` 6장

## 앙상블 시 참고
- 우리 쪽 실험에서는 BCE 모델끼리 평균을 내면 0.5 고정 판정에서 단일 모델보다 낮았다. 평균을 내면 확률이 가운데로 모여 0.5를 넘는 증상이 줄고 빈 예측이 늘기 때문이다 (KLUE + DeBERTa + mmBERT 최적 가중치가 0.9 : 0 : 0.1).
- `klue_sj1` 확률은 soft Jaccard로 학습해서 BCE 모델보다 확신이 강하게 나온다. 다른 모델과 가중치를 탐색할 때 이 차이가 반영된다.
