Mission 3 - klue_roberta_large 단독 결과 (2번 후보)
============================================================

체크포인트: klue_roberta_large_posweight_selective (power=1.0)
단독 성능: macro_f1 0.6328, micro_f1 (final_eval_metrics.json 참고)

클래스 순서 (공식 9개 순서, 모든 csv 동일): ['고열', '구토', '두통', '복통', '어지러움', '열상', '오심', '전신쇠약', '호흡곤란']

Threshold: 0.5 공통 고정 (대회 규칙상 클래스별로 다르게 쓴 적 없음)

klue_roberta_large_val_probs.csv: label_file + 9개 증상 sigmoid probability(0~1, thresholding 전)
val_ground_truth_labels.csv: label_file + 9개 증상 정답 라벨(0/1), split 일치 확인용
두 csv 모두 label_file로 조인 가능 (같은 mission3_val.csv 기준 고정 split)
