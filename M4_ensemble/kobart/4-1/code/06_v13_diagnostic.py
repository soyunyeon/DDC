import csv
import json
from pathlib import Path

root = Path("Results/M4_1")

files = {
    "v11B": root / "t5_t5_m41_v11b_mix50_3000_val.csv",
    "v13A": root / "t5_t5_m41_v13a_targeted_3000_val.csv",
    "early10": Path("Results/V13A/early10_val.csv"),
}

prob_path = (
    root / "t5_t5_m41_v13a_targeted_3000_val_probs.jsonl"
)

with open("Work/Common/label_map.json", encoding="utf-8") as f:
    labels = json.load(f)["labels"]

assert len(labels) == 9
label_set = set(labels)

def group(step):
    step = int(step)
    if step <= 3: return "01-03"
    if step <= 6: return "04-06"
    if step <= 10: return "07-10"
    if step <= 20: return "11-20"
    return "21+"

groups = ["01-03", "04-06", "07-10", "11-20", "21+"]

def empty():
    return {
        "tp": [0] * 9,
        "fp": [0] * 9,
        "fn": [0] * 9
    }

stats = {
    name: {g: empty() for g in ["ALL"] + groups}
    for name in files
}

# 정답 데이터 읽기
gold = {}

with prob_path.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        key = (r["label_file"], int(r["step"]))
        assert key not in gold, f"중복 정답: {key}"
        gold[key] = [int(v) for v in r["target_vector"]]

print("정답 로드:", len(gold))

assert len(gold) == 179169

# 세 모델 비교
for name, path in files.items():
    assert path.exists(), f"파일 없음: {path}"

    seen = set()

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            key = (r["label_file"], int(r["step"]))

            if key not in gold:
                raise ValueError(f"{name}: 정답 없는 시점 {key}")

            if key in seen:
                raise ValueError(f"{name}: 중복 예측 {key}")

            seen.add(key)

            # 수정: 공식 CSV 증상 구분자는 세미콜론
            predicted = {
                s.strip()
                for s in r["symptom"].split(";")
                if s.strip()
            }

            unknown = predicted - label_set
            if unknown:
                raise ValueError(
                    f"{name}: 알 수 없는 증상 {unknown}, "
                    f"원본={r['symptom']!r}"
                )

            y = gold[key]
            g = group(key[1])

            for target in ["ALL", g]:
                s = stats[name][target]

                for i, label in enumerate(labels):
                    p = label in predicted
                    actual = y[i] == 1

                    s["tp"][i] += int(p and actual)
                    s["fp"][i] += int(p and not actual)
                    s["fn"][i] += int(not p and actual)

    assert seen == set(gold), f"{name}: 평가 시점 불일치"

    print(f"{name}: {len(seen):,}개 확인 완료")

def precision(s, i):
    tp, fp = s["tp"][i], s["fp"][i]
    return tp / max(tp + fp, 1)

def recall(s, i):
    tp, fn = s["tp"][i], s["fn"][i]
    return tp / max(tp + fn, 1)

def f1(s, i):
    tp, fp, fn = s["tp"][i], s["fp"][i], s["fn"][i]
    return 2 * tp / max(2 * tp + fp + fn, 1)

def macro(s):
    return sum(f1(s, i) for i in range(9)) / 9

print("\n===== 증상별 F1 =====")
print("증상       v11B    v13A   Early10   변화")

for i, label in enumerate(labels):
    a = f1(stats["v11B"]["ALL"], i)
    b = f1(stats["v13A"]["ALL"], i)
    c = f1(stats["early10"]["ALL"], i)

    print(
        f"{label:<6} "
        f"{a:.4f}  {b:.4f}   {c:.4f}  {c-a:+.4f}"
    )

print("\n===== v13-A Early10 Precision / Recall =====")
print("증상        P        R        F1")

for i, label in enumerate(labels):
    s = stats["early10"]["ALL"]

    print(
        f"{label:<6} "
        f"{precision(s,i):.4f}   "
        f"{recall(s,i):.4f}   "
        f"{f1(s,i):.4f}"
    )

print("\n===== 발화 구간별 Macro F1 =====")
print("구간       v11B    v13A   Early10")

for g in groups:
    print(
        f"{g:<7} "
        f"{macro(stats['v11B'][g]):.4f}  "
        f"{macro(stats['v13A'][g]):.4f}   "
        f"{macro(stats['early10'][g]):.4f}"
    )

print("\n===== 전체 진단용 Macro F1 =====")

for name in files:
    print(name, f"{macro(stats[name]['ALL']):.4f}")

print("\n분석 완료!")
