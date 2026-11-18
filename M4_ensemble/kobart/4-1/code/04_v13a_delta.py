import csv
import json
import math
from collections import defaultdict
from pathlib import Path

base = Path("Results/M4_1")
csv_path = base / "t5_t5_m41_v13a_targeted_3000_val.csv"
prob_path = base / "t5_t5_m41_v13a_targeted_3000_val_probs.jsonl"

with open("Work/Common/label_map.json", encoding="utf-8") as f:
    labels = json.load(f)["labels"]

assert len(labels) == 9

# 1. 기존 공식 채점용 CSV 읽기
original = {}

with csv_path.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    assert reader.fieldnames == ["label_file", "step", "symptom"]

    for row in reader:
        key = (row["label_file"], int(row["step"]))
        assert key not in original
        original[key] = row["symptom"]

# 2. 기존 모델 확률 읽기
probs = {}
calls = defaultdict(list)

with prob_path.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        key = (r["label_file"], int(r["step"]))
        p = [float(x) for x in r["probs"]]

        assert len(p) == 9
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in p)
        assert key not in probs

        probs[key] = p
        calls[key[0]].append((key[1], p))

assert set(original) == set(probs)
assert len(original) == 179169

print("데이터 정합성 확인:", len(original), "개")

# 3. 기존 CSV의 증상 직렬화 형식 자동 탐색
styles = []

for sep in [",", ", ", "|", " | ", ";", "; ", " ", "/", " / ", "、"]:
    for sort_names in [False, True]:
        styles.append(("join", sep, sort_names))

styles += [
    ("json", "", False),
    ("python", "", False),
]

def serialize(names, style):
    mode, sep, sort_names = style

    if sort_names:
        names = sorted(names)

    if mode == "json":
        return json.dumps(names, ensure_ascii=False)
    if mode == "python":
        return str(names)
    return sep.join(names)

sample_keys = list(original)[:20000]
best_style = None
best_matches = -1

for style in styles:
    matches = 0

    for key in sample_keys:
        names = [
            labels[i] for i, p in enumerate(probs[key])
            if p >= 0.5
        ]

        if serialize(names, style) == original[key]:
            matches += 1

    if matches > best_matches:
        best_matches = matches
        best_style = style

print("탐색한 CSV 형식:", best_style)
print("샘플 일치:", best_matches, "/", len(sample_keys))

# 전체 기존 CSV를 동일하게 재현할 수 있는지 검사
for key, expected in original.items():
    names = [
        labels[i] for i, p in enumerate(probs[key])
        if p >= 0.5
    ]
    actual = serialize(names, best_style)

    if actual != expected:
        raise RuntimeError(
            f"기존 CSV 재현 실패: {key}\n"
            f"기존: {expected!r}\n"
            f"재현: {actual!r}\n"
            "출력 형식을 먼저 확인해야 합니다."
        )

print("기존 v11-B CSV 100% 재현 성공!")

# 4. 각 통화 내에서 과거 -> 현재 순서로 확률 변화량 계산
# 첫 발화에서는 변화량 0
deltas = {}

for label_file, rows in calls.items():
    rows.sort(key=lambda x: x[0])
    previous = None

    for step, current in rows:
        if previous is None:
            delta = [0.0] * 9
        else:
            delta = [
                current[i] - previous[i]
                for i in range(9)
            ]

        deltas[(label_file, step)] = delta
        previous = current

assert len(deltas) == len(original)

# 5. 실험별 보정 예측 생성
alphas = [0.25, 0.50, 1.00]

for alpha in alphas:
    name = f"t5_v13a_delta_a{round(alpha*100):03d}_val.csv"
    output = base / name

    if output.exists():
        raise FileExistsError(f"기존 파일 보호: {output}")

    changed = 0

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label_file", "step", "symptom"]
        )
        writer.writeheader()

        for (label_file, step), old_symptom in original.items():
            key = (label_file, step)

            calibrated = [
                min(1.0, max(0.0, p + alpha * d))
                for p, d in zip(probs[key], deltas[key])
            ]

            predicted = [
                labels[i]
                for i, p in enumerate(calibrated)
                if p >= 0.5
            ]

            symptom = serialize(predicted, best_style)
            changed += int(symptom != old_symptom)

            writer.writerow({
                "label_file": label_file,
                "step": step,
                "symptom": symptom
            })

    print(
        f"alpha={alpha:.2f} | "
        f"변경된 시점={changed:,} | "
        f"CSV={output}"
    )

print("\nV12-B 예측 생성 완료!")
