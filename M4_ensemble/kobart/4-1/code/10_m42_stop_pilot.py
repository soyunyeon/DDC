import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

ROOT = Path("Results")
INPUT = ROOT / "V13A/early10_val.csv"
OUTPUT = ROOT / "M42_Pilot"

assert INPUT.is_file(), f"입력 파일 없음: {INPUT}"

calls = defaultdict(list)
total = 0

# 4-1 최종 예측 불러오기
with INPUT.open(
    encoding="utf-8-sig",
    newline=""
) as f:
    reader = csv.DictReader(f)

    required = {"label_file", "step", "symptom"}
    assert required.issubset(reader.fieldnames or [])

    for row in reader:
        lf = row["label_file"]
        step = int(row["step"])

        calls[lf].append({
            "step": step,
            "symptom": row["symptom"]
        })
        total += 1

assert len(calls) == 5840, len(calls)
assert total == 179169, total

# 발화 순서 검사
for lf, rows in calls.items():
    rows.sort(key=lambda r: r["step"])

    steps = [r["step"] for r in rows]
    assert steps == list(range(1, len(rows) + 1)), lf

print("===== 데이터 확인 =====")
print("통화 수:", len(calls))
print("전체 Prefix:", total)

# 우선 고정 중단 기준부터 검증
for name, cutoff in [
    ("fixed5", 5),
    ("fixed10", 10)
]:
    path = OUTPUT / f"{name}.csv"
    halt_steps = []
    nonempty = 0

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "label_file",
            "halt_step",
            "symptom"
        ])

        for lf, rows in calls.items():

            # 시간순 입력을 받는다고 가정
            chosen = None

            for row in rows:
                if row["step"] >= cutoff:
                    chosen = row
                    break

            # 해당 시점에 도달하기 전 통화가 종료된 경우
            if chosen is None:
                chosen = rows[-1]

            writer.writerow([
                lf,
                chosen["step"],
                chosen["symptom"]
            ])

            halt_steps.append(chosen["step"])

            if chosen["symptom"].strip():
                nonempty += 1

    print(f"\n===== {name} =====")
    print("저장:", path)
    print("예측 통화:", len(halt_steps))
    print("평균 중단:", round(mean(halt_steps), 2))
    print("중앙값:", median(halt_steps))
    print("증상 출력 통화:", nonempty)

print("\nCSV 생성 완료!")
