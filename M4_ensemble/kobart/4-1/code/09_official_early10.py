import csv
from pathlib import Path
from itertools import zip_longest

BASE = Path(
    "Results/M4_1/"
    "t5_t5_m41_v13a_officialval_val.csv"
)

DELTA = Path(
    "Results/M4_1/"
    "t5_v13a_official_delta_a100_val.csv"
)

OUTPUT = Path(
    "Results/OfficialVal/v13a_early10_final.csv"
)

assert BASE.is_file(), "원본 예측 없음"
assert DELTA.is_file(), "Delta 보정 예측 없음"
assert not OUTPUT.exists(), "최종 파일이 이미 존재합니다."

fields = ["label_file", "step", "symptom"]

count = 0
changed = 0
calls = set()

with (
    BASE.open(encoding="utf-8-sig", newline="") as fa,
    DELTA.open(encoding="utf-8-sig", newline="") as fb,
    OUTPUT.open("w", encoding="utf-8", newline="") as fo
):
    ra = csv.DictReader(fa)
    rb = csv.DictReader(fb)

    assert ra.fieldnames == fields
    assert rb.fieldnames == fields

    writer = csv.DictWriter(fo, fieldnames=fields)
    writer.writeheader()

    for a, b in zip_longest(ra, rb):
        assert a is not None and b is not None
        assert a["label_file"] == b["label_file"]
        assert a["step"] == b["step"]

        step = int(a["step"])

        # 기존 Early10과 완전히 동일한 선택 규칙
        if step <= 10:
            chosen = b
        else:
            chosen = a

        changed += int(
            chosen["symptom"] != a["symptom"]
        )

        writer.writerow(chosen)

        calls.add(a["label_file"])
        count += 1

assert count == 111947, count
assert len(calls) == 3640, len(calls)

print("===== 공식 Validation Early10 완료 =====")
print("전체 통화:", len(calls))
print("전체 Prefix:", count)
print("변경 예측:", changed)
print("최종 CSV:", OUTPUT)
