import json
from pathlib import Path
from collections import Counter

ROOT = Path("/data1/workspaces/ddc/DCC_Final")

TRAIN_PATH = ROOT / "Work/Common/mission4_train.jsonl"
VAL_PATH = ROOT / "Work/Common/mission4_val.jsonl"

OUT_DIR = ROOT / "Work/Qwen3_LoRA/Data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUT_DIR / "prefix_train_8.jsonl"
VAL_OUT = OUT_DIR / "prefix_val_all.jsonl"


def select_train_steps(n):
    """초기 발화를 강조하면서 최대 8개 시점을 선택."""
    if n <= 8:
        return list(range(1, n + 1))

    steps = {1, 2, 3, 5, 8, 12, n}

    # 나머지 중간 시점 하나 추가
    steps.add(max(1, round(n * 0.7)))

    steps = {s for s in steps if 1 <= s <= n}

    return sorted(steps)


def format_utterance(utt):
    """화자 정보와 텍스트를 하나의 발화 문자열로 변환."""
    speaker = "신고자" if utt["speaker"] == 1 else "119대원"
    return f"{speaker}: {utt['text']}"


def process(input_path, output_path, mode):
    total_calls = 0
    total_prefixes = 0
    step_counts = Counter()

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            row = json.loads(line)

            utterances = row["utterances"]
            n = len(utterances)

            if n == 0:
                continue

            if mode == "train":
                steps = select_train_steps(n)
            else:
                steps = list(range(1, n + 1))

            formatted = [format_utterance(u) for u in utterances]

            for step in steps:
                prefix_text = "\n".join(formatted[:step])

                record = {
                    "call_id": row["call_id"],
                    "label_file": row["label_file"],
                    "step": step,
                    "text": prefix_text,
                    "label_vector": row["label_vector"],
                    "symptom": row["symptom"],
                }

                fout.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                total_prefixes += 1

            total_calls += 1
            step_counts[len(steps)] += 1

            if total_calls % 2000 == 0:
                print(
                    f"[{mode}] {total_calls:,} calls / "
                    f"{total_prefixes:,} prefixes",
                    flush=True
                )

    print(f"\n[{mode}] 완료")
    print(f"통화 수: {total_calls:,}")
    print(f"Prefix 수: {total_prefixes:,}")
    print(f"저장: {output_path}")

    if mode == "train":
        print("통화별 Prefix 개수 분포:", dict(sorted(step_counts.items())))


def main():
    print("=" * 60)
    print("Mission 4 - Qwen3 Prefix Dataset")
    print("=" * 60)

    process(TRAIN_PATH, TRAIN_OUT, "train")
    process(VAL_PATH, VAL_OUT, "val")

    print("\n전체 데이터 생성 완료")


if __name__ == "__main__":
    main()
