checkpoint는 아래링크에서 다운로드.
https://drive.google.com/file/d/1ShzVaJu9P7_6ETrJiJdEMq12UJ0fay7q/view?usp=sharing

## 실행

```
python inference.py --audio_dir {wav folder} --label_dir {json folder} --ckpt_path ./checkpoints/best_model.pt --output ./outputs/mission3.csv
```

- 인터넷 없이 실행된다. tokenizer와 config는 `assets/`에서 읽고, encoder 가중치는 체크포인트에 모두 들어 있다.
- `--audio_dir`은 받기만 하고 쓰지 않는다. Mission 3는 텍스트만 입력으로 쓴다.
- JSON에서는 `utterances[].text`만 읽는다. `symptom` 등 다른 항목은 읽지 않는다.
- Decision Threshold는 0.5로 고정되어 있다.
- 선택 옵션으로 `--prob_output {경로}`를 주면 증상별 sigmoid 확률(threshold 적용 전)도 같이 저장한다.

## 출력 형식

`label_file_name, symptom` 두 컬럼의 CSV. `symptom`은 예측된 증상 리스트를 문자열로 저장한다. 예: `['두통', '복통']`. 예측된 증상이 없으면 `[]`.

## 모델 구조

1. 콜을 발화 단위로 나눈다 (콜당 최대 16개, 넘으면 긴 발화 우선).
2. 각 발화를 encoder에 넣고 token-level mean pooling으로 발화 임베딩을 만든다 (발화당 최대 64 token).
3. 발화 임베딩을 콜 단위로 mean pooling과 max pooling해서 이어붙인다.
4. Linear head로 9개 증상 logit을 만들고 sigmoid를 적용한다.

전체 대화를 한 번에 넣는 방식과 달리 512 token 제한으로 대화가 잘리지 않는다.

## 학습 요약

- Backbone: `jhgan/ko-sroberta-multitask`, 마지막 3개 transformer layer와 head만 학습
- Loss: BCEWithLogits, pos_weight = sqrt(N_neg / N_pos) (최대 10)
- Optimizer: AdamW (encoder lr 2e-5, head lr 1e-3), FP16 mixed precision, batch size 8
- Early stopping: dev macro F1 기준 (threshold 0.5), patience 2. epoch 2 체크포인트 선택
- Train 23,360건 / Dev 5,840건 (Training 내부 80/20 split). Validation 데이터는 학습에 쓰지 않았다.

## 성능

| 평가 | Macro F1 (threshold 0.5) |
|---|---|
| 내부 Dev (5,840건) | 0.6265 |
| Official Validation (3,640건) | 0.6321 |

## 계산 효율

- Total parameters: 110,631,945
- Active parameters: 110,631,945 (dense 모델이라 추론 시 전체 파라미터를 사용)
- Trainable parameters (학습 시): 21,277,449
- 학습 및 추론 환경: NVIDIA GeForce RTX 3050 4GB Laptop GPU
- Validation 추론 batch size: 16
- Official Validation 3,640건 전체 추론시간: 748.64초 (샘플당 평균 205.67ms)
