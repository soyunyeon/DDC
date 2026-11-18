import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

# ==================================================
# Mission 4-2
# 연속 예측 횟수 + 중단 시작 시점 비교
# Delta = 0.05 고정
# ==================================================

pred_path = Path("Results/V13A/early10_val.csv")

prob_path = Path(
    "Results/M4_1/"
    "t5_t5_m41_v13a_targeted_3000_val_probs.jsonl"
)

out_dir = Path("Results/M42_Pilot")
out_dir.mkdir(parents=True, exist_ok=True)

assert pred_path.is_file(), pred_path
assert prob_path.is_file(), prob_path

preds = defaultdict(dict)
probs = defaultdict(dict)


def normalize(symptom):
    return tuple(sorted(
        x.strip()
        for x in symptom.split(";")
        if x.strip()
    ))


# ==================================================
# 1. Early10 예측 불러오기
# ==================================================

with pred_path.open(
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    assert {
        "label_file", "step", "symptom"
    }.issubset(reader.fieldnames or [])

    for r in reader:
        lf = r["label_file"]
        step = int(r["step"])

        assert step not in preds[lf]

        preds[lf][step] = r["symptom"]


# ==================================================
# 2. T5 예측 확률 불러오기
# ==================================================

with prob_path.open(encoding="utf-8") as f:

    for line in f:
        r = json.loads(line)

        lf = r["label_file"]
        step = int(r["step"])

        p = [float(x) for x in r["probs"]]

        assert len(p) == 9
        assert all(
            math.isfinite(x) and 0 <= x <= 1
            for x in p
        )

        assert step not in probs[lf]

        probs[lf][step] = p


assert len(preds) == len(probs) == 5840
assert preds.keys() == probs.keys()

for lf in preds:
    expected = set(range(1, len(preds[lf]) + 1))

    assert set(preds[lf]) == expected
    assert set(probs[lf]) == expected

print("데이터 확인 완료")
print("전체 통화:", len(preds))


# ==================================================
# 3. 실험 설정
# ==================================================

configs = [
    (2, 6),
    (2, 8),
    (3, 6),
    (3, 8),
    (4, 6),
    (4, 8),
]

DELTA_THRESHOLD = 0.05


# ==================================================
# 4. 중단 정책 실험
# ==================================================

for n, start in configs:

    name = f"win{n}_start{start}_d050"
    output = out_dir / f"{name}.csv"

    halt_steps = []

    with output.open(
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

        for lf, call_preds in preds.items():

            call_probs = probs[lf]

            # 중단 조건을 만족하지 못하면
            # 실제 스트림 종료 시점에 결과 출력
            chosen_step = len(call_preds)

            for t in range(
                max(n, start),
                len(call_preds) + 1
            ):

                labels = [
                    normalize(call_preds[j])
                    for j in range(t-n+1, t+1)
                ]

                # 빈 예측은 조기 중단하지 않음
                if not labels[-1]:
                    continue

                # N회 연속 동일한 증상인지 확인
                if any(
                    x != labels[-1]
                    for x in labels
                ):
                    continue

                # 최근 N회 예측 확률 변화량
                delta_max = max(
                    abs(a-b)
                    for j in range(t-n+2, t+1)
                    for a, b in zip(
                        call_probs[j-1],
                        call_probs[j]
                    )
                )

                if delta_max <= DELTA_THRESHOLD:
                    chosen_step = t
                    break

            writer.writerow([
                lf,
                chosen_step,
                call_preds[chosen_step]
            ])

            halt_steps.append(chosen_step)

    print("\n" + "=" * 55)
    print("실험:", name)
    print("통화 수:", len(halt_steps))
    print("평균 중단:", round(mean(halt_steps), 2))
    print("중앙값:", median(halt_steps))
    print(
        "10회 이내 중단:",
        f"{sum(x <= 10 for x in halt_steps) / len(halt_steps) * 100:.2f}%"
    )
    print("저장:", output)

print("\n전체 실험 CSV 생성 완료!")
