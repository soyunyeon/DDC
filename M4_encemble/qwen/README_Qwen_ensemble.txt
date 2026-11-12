DCC Mission 4: Qwen ensemble export kit (2026-11-12)

1. Prepare raw input in your Windows machine:
   D:\DCC_Backup\M4_Qwen\val_all_merged.jsonl
   This 53MB file was transferred separately from sw51 and is NOT in
   M4_Qwen_GitHub.tar.gz. If this file is missing, local CSV creation is not possible.

2. Copy prepare_qwen_ensemble.py to:
   D:\DCC_Backup\M4_Qwen\prepare_qwen_ensemble.py

3. Windows PowerShell (no GPU or server needed):
   & "D:\DCC\.venv\Scripts\python.exe" "D:\DCC_Backup\M4_Qwen\prepare_qwen_ensemble.py"

   Or use your available Python >=3.10:
   py "D:\DCC_Backup\M4_Qwen\prepare_qwen_ensemble.py"

4. Output, all saved to D:\DCC_Backup\M4_Qwen\:
   val_holdout_probs.csv              M4-1 9-class probabilities, each prefix
   val_holdout_m41_pred.csv           M4-1 predictions (submission format)
   val_holdout_m42_pred.csv           M4-2 stop step + symptom list (submission format)
   val_holdout_m42_details.csv        M4-2 stop step + nine 0/1 symptom columns
   val_holdout_m42_internal_score.json  Internal Validation proxy score

5. The team must confirm val_holdout == the original Common/mission4_val.jsonl.
   This script labels the output 'val_holdout' by request; it cannot independently
   establish that the team uses the same call_id/step keys.

6. label_file is reconstructed as <call_id>.json, matching the original team
   01_common_preprocess.py rule: call_id = Path(label_file).stem.
   Optional (preferable when available): provide exact team split manifest:
   ...python prepare_qwen_ensemble.py --manifest "path/to/split_manifest.csv"

7. M4-2 stopping (k2_c70_cap28): symptom threshold=0.5, min step=5,
   2 consecutive equal nonempty label sets, peak confidence >=0.7,
   hard cap step 28, or stream end. It never reads future utterances.

8. Additional included files:
   inference_qwen_m42.py: Predictor draft developed previously; syntax/mock-tested
     but NOT certified by an end-to-end submission run.
   m42_stop_policy.py: lightweight streaming policy, no model dependencies.
   requirements_qwen.txt: representative Qwen environment (not full reproducibility lock).

9. Weights and original code are NOT in this kit. They remain in the separate
   M4_Qwen_GitHub.tar.gz backup (adapter and Code/).
   Qwen3-4B base weights and raw AIHub datasets are excluded. Do not claim the
   kit alone suffices to execute the Qwen model in a new environment.
