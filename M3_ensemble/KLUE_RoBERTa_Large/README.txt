Mission 3 - klue_roberta_large 단독 결과 (2번 후보)
============================================================

체크포인트: klue_roberta_large_posweight_selective (power=1.0)
단독 성능: macro_f1 0.6328, micro_f1 (final_eval_metrics.json 참고)

클래스 순서 (공식 9개 순서, 모든 csv 동일): ['고열', '구토', '두통', '복통', '어지러움', '열상', '오심', '전신쇠약', '호흡곤란']

Threshold: 0.5 공통 고정 (대회 규칙상 클래스별로 다르게 쓴 적 없음)

klue_roberta_large_val_probs.csv: label_file + 9개 증상 sigmoid probability(0~1, thresholding 전)
val_ground_truth_labels.csv: label_file + 9개 증상 정답 라벨(0/1), split 일치 확인용
두 csv 모두 label_file로 조인 가능 (같은 mission3_val.csv 기준 고정 split)

============================================================
체크포인트 / 코드 (추가, 2026-10-11)
============================================================

이 폴더에 있는 것: config.json, tokenizer.json, tokenizer_config.json, inference.py

model.safetensors(학습된 가중치, 약 1.25GB)는 GitHub에 올리지 않았습니다.
이미 mission3_best/best_model.pt(420MB)가 LFS로 올라가 있어서 여기에
1.25GB를 더 추가하면 GitHub 무료 LFS 저장용량(1GB)을 초과합니다.

model.safetensors 다운로드:
https://drive.google.com/drive/folders/1u6z8TnLQJd2PMjR8goO7ksYBW81nLHmO?usp=sharing

사용 방법:
1. 위 Drive 링크에서 model.safetensors를 받아서, 이 폴더(config.json/
   tokenizer.json/tokenizer_config.json이 있는 폴더)에 같이 넣는다.
   (즉 model.safetensors + config.json + tokenizer.json +
   tokenizer_config.json 4개가 한 폴더 안에 모여야 함 - HuggingFace
   save_pretrained 형식이라 폴더 전체를 하나의 체크포인트로 취급함)
2. 아래처럼 inference.py 실행:

   python inference.py \
       --label_dir <라벨 json 폴더> \
       --ckpt_path <위 폴더 경로> \
       --output ./outputs/mission3.csv

   --ckpt_path는 단일 파일이 아니라 폴더 경로입니다.
