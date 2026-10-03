# Mission 3 Best Model Share

## Best Model

- Experiment: 08 Truncation Baseline
- Backbone: KLUE-RoBERTa-base
- Internal Dev Macro F1: 0.621428
- Decision Threshold: 0.5

## Input

Only `utterances[].text` is used.

The following annotation fields are NOT used:

- speaker
- startAt
- endAt
- gender
- triage
- urgency
- address
- other annotation fields

## Class Order

1. 고열
2. 구토
3. 두통
4. 복통
5. 어지러움
6. 열상
7. 오심
8. 전신쇠약
9. 호흡곤란

## Model Configuration

- max_length: 512
- chunking: False
- chunk_size: None
- stride: None
- max_chunks: 1
- pooling: CLS
- attention_pooling: False
- loss: BCEWithLogitsLoss
- threshold: 0.5
- seed: 42

## Internal Split

- Train: 23,348
- Dev: 5,852

## Files

- internal_dev_probs.csv
  - label_file_name + 9 sigmoid probabilities

- internal_dev_labels.csv
  - label_file_name + 9 multi-hot ground-truth labels

- model_config.json
  - class order / threshold / model configuration

- best_model.pt
  - best checkpoint

- 01_common_preprocess.py
  - preprocessing

- 08_train_truncation_baseline.py
  - training / model definition

- inference.py
  - inference code
