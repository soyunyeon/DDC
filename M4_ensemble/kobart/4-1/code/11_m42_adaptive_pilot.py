import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

pred_path = Path("Results/V13A/early10_val.csv")
prob_path = Path(
    "Results/M4_1/"
    "t5_t5_m41_v13a_targeted_3000_val_probs.jsonl"
)
out_dir = Path("Results/M42_Pilot")

assert pred_path.is_file()
assert prob_path.is_file()
out_dir.mkdir(parents=True, exist_ok=True)

preds = defaultdict(dict)
probs = defaultdict(dict)

with pred_path.open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        key = r["label_file"]
        step = int(r["step"])
        assert step not in preds[key]
        preds[key][step] = r["symptom"]

with prob_path.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        key = r["label_file"]
        step = int(r["step"])
        p = [float(x) for x in r["probs"]]
        assert len(p) == 9
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in p)
        assert step not in probs[key]
        probs[key][step] = p

assert len(preds) == len(probs) == 5840
assert preds.keys() == probs.keys()

def normalize(s):
    return tuple(sorted(
        x.strip() for x in s.split(";") if x.strip()
    ))

policies = {
    "stable3_from6": None,
    "stable3_delta05": 0.05,
    "stable3_delta03": 0.03,
    "stable3_delta035": 0.035,
    "stable3_delta040": 0.040,
    "stable3_delta045": 0.045,
    "stable3_delta055": 0.055,
    "stable3_delta060": 0.060,
    "stable3_delta070": 0.070,
}

for name, limit in policies.items():
    output = out_dir / f"{name}.csv"
    halt_steps = []

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_file", "halt_step", "symptom"])

        for lf, call_preds in preds.items():
            call_probs = probs[lf]

            steps = sorted(call_preds)
            assert steps == sorted(call_probs)
            assert steps == list(range(1, len(steps) + 1))

            # 실제 스트림 종료 시 마지막 예측 반환
            chosen_step = steps[-1]

            for t in steps:
                if t < 6:
                    continue

                a = normalize(call_preds[t-2])
                b = normalize(call_preds[t-1])
                c = normalize(call_preds[t])

                # 빈 증상 출력으로 조기 중단하지 않음
                if not c or not (a == b == c):
                    continue

                if limit is not None:
                    p0 = call_probs[t-2]
                    p1 = call_probs[t-1]
                    p2 = call_probs[t]

                    change = max(
                        max(abs(x-y) for x,y in zip(p0,p1)),
                        max(abs(x-y) for x,y in zip(p1,p2)),
                    )

                    if change > limit:
                        continue

                chosen_step = t
                break

            writer.writerow([
                lf,
                chosen_step,
                call_preds[chosen_step]
            ])

            halt_steps.append(chosen_step)

    print(f"\n===== {name} =====")
    print("통화 수:", len(halt_steps))
    print("평균 중단:", round(mean(halt_steps), 2))
    print("중앙값:", median(halt_steps))
    print("10회 이내 중단:",
          f"{sum(x<=10 for x in halt_steps)/len(halt_steps)*100:.2f}%")
    print("CSV:", output)

print("\n실험 CSV 생성 완료")
