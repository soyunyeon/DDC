import csv
import json
import math
import ast
import re
from pathlib import Path

root = Path("Results/M4_1")

files = {
    "v13": root / "t5_t5_m41_v13a_targeted_3000_val_probs.jsonl",
    "v14": root / "t5_t5_m41_v14a_contrast_2000_val_probs.jsonl",
}

base_path = Path("Results/V13A/early10_val.csv")
label_path = Path("Work/Common/label_map.json")

for name, p in files.items():
    if not p.is_file():
        candidates = list(Path("Results").rglob(
            f"*{'v13a' if name == 'v13' else 'v14a'}*probs.jsonl"
        ))
        if len(candidates) == 1:
            files[name] = candidates[0]
        else:
            raise FileNotFoundError(
                f"{name} 확률 파일을 특정할 수 없음: {candidates}"
            )

assert base_path.is_file()
with label_path.open(encoding="utf-8") as f:
    labels = json.load(f)["labels"]

assert len(labels) == 9

def load_probs(path):
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["label_file"], int(r["step"]))
            assert key not in result
            p = [float(x) for x in r["probs"]]
            assert len(p) == 9
            assert all(math.isfinite(x) and 0 <= x <= 1 for x in p)
            result[key] = p
    return result

v13 = load_probs(files["v13"])
v14 = load_probs(files["v14"])

with base_path.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    assert fields == ["label_file", "step", "symptom"]
    base = list(reader)

keys = [(r["label_file"], int(r["step"])) for r in base]

assert len(keys) == len(set(keys)) == 179169
assert set(keys) == set(v13) == set(v14)

# 기존 CSV의 증상 표현 형식 유지
samples = [
    r["symptom"].strip()
    for r in base if r["symptom"].strip()
]
assert samples

def parse_symptom(s):
    if not s or s == "[]":
        return []
    if s.startswith("["):
        try:
            return list(ast.literal_eval(s))
        except (SyntaxError, ValueError):
            pass
    return [x.strip() for x in re.split(r"[,;|/]", s) if x.strip()]

for s in samples[:100]:
    assert set(parse_symptom(s)) <= set(labels)

multi = next(
    (s for s in samples if len(parse_symptom(s)) > 1),
    samples[0]
)

if multi.startswith("["):
    fmt = "python_list" if "'" in multi else "json_list"
elif "|" in multi:
    fmt = "pipe"
elif ";" in multi:
    fmt = "semicolon"
else:
    fmt = "comma"

def encode(indices):
    names = [labels[i] for i in indices]
    if not names:
        return ""
    if fmt == "python_list":
        return str(names)
    if fmt == "json_list":
        return json.dumps(names, ensure_ascii=False)
    return {"pipe": "|", "semicolon": ";", "comma": ","}.get(
        fmt, ","
    ).join(names)

print("확률 파일:", files)
print("CSV 증상 형식:", fmt)
print("기준 예측:", len(base))

for alpha in [0.25, 0.50]:
    name = f"v15_early_blend_a{int(alpha*100):03d}"
    out = Path("Results/V15") / f"{name}_val.csv"
    assert not out.exists(), f"기존 결과 보호: {out}"

    changed = 0

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row, key in zip(base, keys):
            new_row = dict(row)

            if key[1] <= 10:
                probs = [
                    (1-alpha)*a + alpha*b
                    for a, b in zip(v13[key], v14[key])
                ]

                indices = [
                    i for i, p in enumerate(probs) if p >= 0.5
                ]
                new_symptoms = encode(indices)

                if set(parse_symptom(new_symptoms)) != set(
                    parse_symptom(row["symptom"])
                ):
                    changed += 1

                new_row["symptom"] = new_symptoms

            writer.writerow(new_row)

    print(
        f"{name}: 변경 {changed:,}개 / {len(base):,}개"
    )
    print("저장:", out)

print("V15 앙상블 CSV 생성 완료")
