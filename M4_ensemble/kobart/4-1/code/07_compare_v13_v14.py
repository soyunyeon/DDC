import csv
import json
import re
import ast
from pathlib import Path

ROOT = Path(".")

GOLD_PATH = ROOT / "Work/Seq2Seq/val_prefix.jsonl"
LABEL_PATH = ROOT / "Work/Common/label_map.json"

FILES = {
    "v13": ROOT / "Results/M4_1/t5_t5_m41_v13a_targeted_3000_val.csv",
    "v14": ROOT / "Results/M4_1/t5_t5_m41_v14a_contrast_2000_val.csv",
}

with LABEL_PATH.open(encoding="utf-8") as f:
    labels = json.load(f)["labels"]

assert len(labels) == 9
assert len(set(labels)) == 9

# 정답 불러오기
gold = {}

with GOLD_PATH.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)

        key = (r["label_file"], int(r["step"]))
        assert key not in gold

        y = tuple(int(v) for v in r["target_vector"])
        assert len(y) == 9

        gold[key] = y

print("정답 Prefix:", len(gold))

# CSV symptom 문자열 해석
def parse_symptom(raw):
    raw = (raw or "").strip()

    if not raw or raw.lower() in ("none", "nan", "[]"):
        return set()

    # 리스트 문자열도 지원
    if raw.startswith("[") and raw.endswith("]"):
        try:
            value = ast.literal_eval(raw)
            if isinstance(value, list):
                parts = [str(x).strip() for x in value]
            else:
                parts = re.split(r"[,;|/]", raw)
        except (ValueError, SyntaxError):
            parts = re.split(r"[,;|/]", raw)
    else:
        parts = re.split(r"[,;|/]", raw)

    parts = [
        x.strip().strip("[]'\" ")
        for x in parts
        if x.strip()
    ]

    unknown = set(parts) - set(labels)
    assert not unknown, f"인식되지 않은 증상: {unknown}"

    return {labels.index(x) for x in parts}

predictions = {}

for name, path in FILES.items():
    assert path.is_file(), f"파일 없음: {path}"

    pred = {}

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        assert {"label_file", "step", "symptom"} <= set(reader.fieldnames)

        for r in reader:
            key = (r["label_file"], int(r["step"]))
            assert key not in pred, f"중복 예측: {key}"

            pred[key] = parse_symptom(r["symptom"])

    assert set(pred) == set(gold), (
        f"{name}: 정답과 예측의 키가 다릅니다."
    )

    predictions[name] = pred
    print(f"{name} 예측:", len(pred))

def metrics(keys, model):
    tp = [0] * 9
    fp = [0] * 9
    fn = [0] * 9
    predicted = [0] * 9

    pred = predictions[model]

    for key in keys:
        y = gold[key]
        p = pred[key]

        for i in range(9):
            actual = y[i] == 1
            estimate = i in p

            if estimate:
                predicted[i] += 1

            if actual and estimate:
                tp[i] += 1
            elif not actual and estimate:
                fp[i] += 1
            elif actual and not estimate:
                fn[i] += 1

    result = []

    for i in range(9):
        precision = tp[i] / max(tp[i] + fp[i], 1)
        recall = tp[i] / max(tp[i] + fn[i], 1)

        f1 = (
            2 * tp[i] / (2 * tp[i] + fp[i] + fn[i])
            if 2 * tp[i] + fp[i] + fn[i] > 0
            else 0
        )

        result.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp[i],
            "fp": fp[i],
            "fn": fn[i],
            "pred_ratio": predicted[i] / len(keys),
        })

    return result

keys = list(gold)
old = metrics(keys, "v13")
new = metrics(keys, "v14")

print("\n" + "=" * 85)
print("1. 증상별 F1 비교")
print("=" * 85)

print(
    f"{'증상':<10} {'v13 F1':>9} {'v14 F1':>9} "
    f"{'차이':>10} {'v13 예측%':>11} {'v14 예측%':>11}"
)

for i, label in enumerate(labels):
    diff = new[i]["f1"] - old[i]["f1"]

    print(
        f"{label:<10} "
        f"{old[i]['f1']:>9.4f} "
        f"{new[i]['f1']:>9.4f} "
        f"{diff:>+10.4f} "
        f"{old[i]['pred_ratio']*100:>10.2f}% "
        f"{new[i]['pred_ratio']*100:>10.2f}%"
    )

print("\n" + "=" * 85)
print("2. Precision / Recall 변화")
print("=" * 85)

for i, label in enumerate(labels):
    print(
        f"{label:<10} "
        f"P: {old[i]['precision']:.4f} → {new[i]['precision']:.4f} "
        f"R: {old[i]['recall']:.4f} → {new[i]['recall']:.4f}"
    )

def group(step):
    if step <= 3:
        return "01-03"
    if step <= 6:
        return "04-06"
    if step <= 10:
        return "07-10"
    if step <= 20:
        return "11-20"
    return "21+"

print("\n" + "=" * 85)
print("3. 발화 구간별 Macro F1")
print("=" * 85)

for g in ["01-03", "04-06", "07-10", "11-20", "21+"]:
    selected = [k for k in keys if group(k[1]) == g]

    a = metrics(selected, "v13")
    b = metrics(selected, "v14")

    fa = sum(r["f1"] for r in a) / 9
    fb = sum(r["f1"] for r in b) / 9

    print(
        f"{g:<7} N={len(selected):>7,} "
        f"v13={fa:.4f} v14={fb:.4f} "
        f"변화={fb-fa:+.4f}"
    )

print("\n" + "=" * 85)
print("4. 오심 / 구토 혼동 비교")
print("=" * 85)

ni = labels.index("오심")
vi = labels.index("구토")

for model in ["v13", "v14"]:
    pred = predictions[model]

    nausea_fp_with_vomit = 0
    vomit_fp_with_nausea = 0

    for key in keys:
        y = gold[key]
        p = pred[key]

        if y[ni] == 0 and y[vi] == 1 and ni in p:
            nausea_fp_with_vomit += 1

        if y[vi] == 0 and y[ni] == 1 and vi in p:
            vomit_fp_with_nausea += 1

    print(f"\n[{model}]")
    print("구토 양성 중 오심 오탐:", nausea_fp_with_vomit)
    print("오심 양성 중 구토 오탐:", vomit_fp_with_nausea)
    print("오심 전체 FP:", old[ni]["fp"] if model == "v13" else new[ni]["fp"])
    print("구토 전체 FP:", old[vi]["fp"] if model == "v13" else new[vi]["fp"])

print("\n" + "=" * 85)
print("5. 전체 Prefix 기준 진단용 Macro F1")
print("=" * 85)

for model, results in [("v13", old), ("v14", new)]:
    print(model, f"{sum(r['f1'] for r in results)/9:.4f}")

print("\n분석 완료!")
