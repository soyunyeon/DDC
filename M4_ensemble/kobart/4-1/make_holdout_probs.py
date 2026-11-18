import argparse
import csv
import json
from pathlib import Path

from predictor import Predictor, OUTPUT_LABELS


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    if not src.is_file():
        raise FileNotFoundError(src)

    if dst.exists():
        raise FileExistsError(
            f"기존 결과 보호: {dst}"
        )

    if args.batch < 1:
        raise ValueError("batch는 1 이상이어야 합니다.")

    dst.parent.mkdir(parents=True, exist_ok=True)

    predictor = Predictor(device=args.device)

    fields = ["label_file", "step"] + OUTPUT_LABELS
    temp = Path(str(dst) + ".tmp")

    if temp.exists():
        raise FileExistsError(temp)

    seen = set()
    total = 0

    def write_batch(writer, batch):
        nonlocal total

        texts = [r["input_text"] for r in batch]
        probs_batch = predictor.predict_batch(texts)

        for row, probs in zip(batch, probs_batch):
            result = {
                "label_file": row["label_file"],
                "step": int(row["step"])
            }

            result.update(zip(OUTPUT_LABELS, probs))
            writer.writerow(result)
            total += 1

    try:
        with (
            src.open(encoding="utf-8") as f,
            temp.open("x", encoding="utf-8", newline="") as out
        ):
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()

            batch = []

            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                row = json.loads(line)

                required = {
                    "label_file", "step", "input_text"
                }

                if not required.issubset(row):
                    raise ValueError(
                        f"{lineno}행: 필수 컬럼 부족. "
                        f"현재 컬럼={list(row.keys())}"
                    )

                key = (
                    row["label_file"],
                    int(row["step"])
                )

                if key in seen:
                    raise ValueError(
                        f"중복된 label_file+step: {key}"
                    )

                seen.add(key)

                if not isinstance(row["input_text"], str):
                    raise ValueError(
                        f"{lineno}행 input_text가 문자열이 아닙니다."
                    )

                batch.append(row)

                if len(batch) >= args.batch:
                    write_batch(writer, batch)
                    batch.clear()

                    if total % 5000 == 0:
                        print(
                            f"[진행] {total:,} prefixes",
                            flush=True
                        )

            if batch:
                write_batch(writer, batch)

        if total == 0:
            raise ValueError("입력 데이터가 비어 있습니다.")

        temp.replace(dst)

    except Exception:
        if temp.exists():
            temp.unlink()
        raise

    print("=" * 60)
    print("KoBART v6 holdout 추론 완료")
    print("전체 Prefix:", f"{total:,}")
    print("저장:", dst)
    print("컬럼:", fields)
    print("=" * 60)


if __name__ == "__main__":
    main()
