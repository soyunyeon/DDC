import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "Work" / "Common"
OUT = ROOT / "Work" / "Seq2Seq"

OUT.mkdir(parents=True, exist_ok=True)

with open(COMMON / "label_map.json", encoding="utf-8") as f:
    LABELS = json.load(f)["labels"]

assert len(LABELS) == 9

def select_train_steps(n):
    """초기 발화 강화 + 후반 시점 일부 샘플링."""
    steps = set(range(1, min(n, 8) + 1))

    later = [10, 12, 16, 20, 24, 32, 40, 48,
             64, 80, 100, 128, 160]

    steps.update(k for k in later if k <= n)
    steps.add(n)

    if len(steps) > 12:
        early = sorted(k for k in steps if k <= 8)
        late = sorted(k for k in steps if k > 8)

        slots = 12 - len(early)
        indices = [
            round(i * (len(late) - 1) / (slots - 1))
            for i in range(slots)
        ]
        steps = set(early + [late[i] for i in indices])

    return steps


def speaker_name(value):
    if value == 1:
        return "신고자"
    if value == 0:
        return "119대원"
    return "화자"


def process(split):
    src = COMMON / f"mission4_{split}.jsonl"
    dst = OUT / f"{split}_prefix.jsonl"

    if dst.exists():
        raise FileExistsError(
            f"{dst} 파일이 이미 존재합니다. "
            "기존 실험 결과를 보호하기 위해 중단합니다."
        )

    n_calls = 0
    n_prefixes = 0

    with src.open(encoding="utf-8") as fin, \
         dst.open("w", encoding="utf-8") as fout:

        for line in fin:
            row = json.loads(line)
            assert row["split"] == split

            utts = row["utterances"]
            n = len(utts)
            assert n > 0

            symptoms = [
                c for c in LABELS
                if c in set(row["symptom"])
            ]

            assert symptoms, "빈 증상 라벨"
            assert len(row["label_vector"]) == 9

            target_vector = [
                int(c in symptoms) for c in LABELS
            ]
            assert target_vector == row["label_vector"]

            if split == "train":
                selected = select_train_steps(n)
            else:
                selected = set(range(1, n + 1))

            history = []

            for k, utt in enumerate(utts, start=1):
                speaker = speaker_name(utt.get("speaker"))
                text = str(utt.get("text", "")).strip()

                if not text:
                    text = "[빈 발화]"

                history.append(f"{speaker}: {text}")

                if k not in selected:
                    continue

                record = {
                    "call_id": row["call_id"],
                    "label_file": row["label_file"],
                    "step": k,
                    "input_text": (
                        "대화의 현재 증상을 판단하세요.\n"
                        + "\n".join(history)
                    ),
                    "target_text": ";".join(symptoms),
                    "target_vector": target_vector,
                }

                fout.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )
                n_prefixes += 1

            n_calls += 1

            if n_calls % 2000 == 0:
                print(
                    f"[{split}] {n_calls:,} calls, "
                    f"{n_prefixes:,} prefixes",
                    flush=True
                )

    print(
        f"[완료] {split}: "
        f"{n_calls:,} calls / {n_prefixes:,} prefixes"
    )

    return {
        "calls": n_calls,
        "prefixes": n_prefixes,
        "path": str(dst)
    }


def main():
    stats_path = OUT / "preprocess_stats.json"

    if stats_path.exists():
        raise FileExistsError(
            "기존 전처리 통계가 존재합니다."
        )

    stats = {
        "train": process("train"),
        "val": process("val"),
        "labels": LABELS,
        "train_sampling": "early8_plus_later_max12",
        "validation_sampling": "all_steps"
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("추가 전처리 완료!")


if __name__ == "__main__":
    main()
