Mission 3 앙상블 공유 파일 (SRoBERTa, Han)
========================================

- sroberta_val_predictions.csv
  label_file_name + 9개 증상별 sigmoid probability (threshold 적용 전)

- class_order.json
  9개 증상 이름과 순서

- official_validation_labels.csv
  Official Validation 3,640건의 실제 symptom 정답.
  Validation JSON에 symptom 필드가 실제로 들어있는 것을 확인해서 추출함.
  학습(backprop)에는 사용하지 않았고, 평가/앙상블 가중치 조정 용도로만 사용.
  각자 이 파일로 진짜 official macro F1을 계산할 수 있음.

- Decision Threshold: 0.5 고정. 클래스별 threshold 아님.
  앙상블도 각 모델 확률을 합친 뒤 마지막에 한 번만 0.5 적용할 것.

- 참고: 이 모델(SRoBERTa, 발화 단위 pooling + 마지막 3개 layer partial fine-tuning)의
  official validation macro F1: 0.6321
