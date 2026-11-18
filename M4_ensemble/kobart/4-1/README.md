# Mission 4-1 — KoBART v6 Ensemble

## Model
- Backbone: KoBART
- Dynamic Soft Jaccard Weight: 0.5 → 8.0
- Training Steps: 4000
- Decision Threshold: 0.5
- Output: 9 independent symptom probabilities

## Holdout
- File: val_holdout_probs.csv
- Rows: 111947
- Join keys: label_file + step
- Identical key set to teammate KoBART_V2 holdout CSV

## Probability Column Order
label_file,step,고열,구토,두통,복통,어지러움,열상,오심,전신쇠약,호흡곤란

## Model Files
- checkpoint/model.pt
- checkpoint/config.json
- checkpoint/metadata.json
- checkpoint/tokenizer files
- code/m4_model.py
- label_map.json
- requirements.txt

## Predictor Interface
predictor.py provides:

- Predictor.reset()
- Predictor.step(input_text)
- Predictor.finalize()
- Predictor.predict_batch(texts)

IMPORTANT:
step(input_text) expects the accumulated, preprocessed
conversation text up to the current utterance.

It does not accept a raw utterance directly.

Raw utterance → input_text conversion must reproduce the
original prefix preparation logic in the included code.

No GRU hidden state is maintained.

## Competition Rules
- Decision threshold fixed at 0.5
- Use only past and current utterances
- Do not use future utterances
- Do not use final conversation length
- Do not modify official score.py

## Note
The holdout CSV is ready for ensemble weight search.
Raw-utterance streaming integration requires matching
the original preprocessing format.
