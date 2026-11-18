import csv
from pathlib import Path
from itertools import zip_longest

base = Path("Results/M4_1")

base_csv = base / "t5_t5_m41_v13a_targeted_3000_val.csv"
delta_csv = base / "t5_v13a_delta_a100_val.csv"

assert base_csv.exists(), f"파일 없음: {base_csv}"
assert delta_csv.exists(), f"파일 없음: {delta_csv}"

experiments = {
    "early3": 3,
    "early6": 6,
    "early10": 10,
}

for name, cutoff in experiments.items():

    output = Path(f"Results/V13A/{name}_val.csv")

    if output.exists():
        print(f"[SKIP] 기존 파일 보존: {output}")
        continue

    count = 0
    changed = 0

    with (
        base_csv.open(encoding="utf-8-sig", newline="") as fa,
        delta_csv.open(encoding="utf-8-sig", newline="") as fb,
        output.open("w", encoding="utf-8", newline="") as fo
    ):
        ra = csv.DictReader(fa)
        rb = csv.DictReader(fb)

        expected = ["label_file", "step", "symptom"]

        assert ra.fieldnames == expected
        assert rb.fieldnames == expected

        writer = csv.DictWriter(fo, fieldnames=expected)
        writer.writeheader()

        for a, b in zip_longest(ra, rb):
            assert a is not None and b is not None
            assert a["label_file"] == b["label_file"]
            assert a["step"] == b["step"]

            step = int(a["step"])

            if step <= cutoff:
                chosen = b
            else:
                chosen = a

            changed += int(
                chosen["symptom"] != a["symptom"]
            )

            writer.writerow(chosen)
            count += 1

    assert count == 179169, f"샘플 수 불일치: {count}"

    print("=" * 55)
    print("실험:", name)
    print("Delta 적용 구간:", f"1~{cutoff}")
    print("전체 예측:", f"{count:,}")
    print("변경된 예측:", f"{changed:,}")
    print("저장 경로:", output)

print("\n모든 실험 CSV 생성 완료!")
