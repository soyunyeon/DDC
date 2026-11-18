import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

# ==========================================
# Mission 4-2
# Confidence + Nausea/Vomiting Guard
# ==========================================

pred_path = Path("Results/V13A/early10_val.csv")

prob_path = Path(
    "Results/M4_1/"
    "t5_t5_m41_v13a_targeted_3000_val_probs.jsonl"
)

out_dir = Path("Results/M42_Pilot")
out_dir.mkdir(parents=True, exist_ok=True)

LABELS = [
    "고열", "구토", "두통",
    "복통", "어지러움", "열상",
    "오심", "전신쇠약", "호흡곤란"
]

IDX = {name: i for i, name in enumerate(LABELS)}

preds = defaultdict(dict)
probs = defaultdict(dict)


def normalize(s):
    labels = tuple(sorted(
        x.strip() for x in s.split(";") if x.strip()
    ))
    assert all(x in IDX for x in labels), labels
    return labels


# ==========================================
# 1. 기존 Early10 예측 읽기
# ==========================================

with pred_path.open(
    encoding="utf-8-sig", newline=""
) as f:
    reader = csv.DictReader(f)

    assert {
        "label_file", "step", "symptom"
    }.issubset(reader.fieldnames or [])

    for r in reader:
        lf = r["label_file"]
        t = int(r["step"])

        assert t not in preds[lf]

        preds[lf][t] = (
            r["symptom"],
            normalize(r["symptom"])
        )


# ==========================================
# 2. 원본 T5 확률 읽기
# ==========================================

with prob_path.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)

        lf = r["label_file"]
        t = int(r["step"])
        p = [float(x) for x in r["probs"]]

        assert len(p) == 9
        assert all(
            math.isfinite(x) and 0 <= x <= 1
            for x in p
        )
        assert t not in probs[lf]

        probs[lf][t] = p


assert len(preds) == len(probs) == 5840
assert preds.keys() == probs.keys()

for lf in preds:
    expected = set(range(1, len(preds[lf]) + 1))

    assert set(preds[lf]) == expected
    assert set(probs[lf]) == expected

print("데이터 확인 완료")
print("전체 통화:", len(preds))


# ==========================================
# 3. 실험 설정
# ==========================================

CONFIGS = [
    ("base_3_d050", None, False),
    ("conf055", 0.55, False),
    ("conf060", 0.60, False),
    ("nv_guard", None, True),
    ("conf055_nv", 0.55, True),
]


# ==========================================
# 4. 실시간 중단 조건
# ==========================================

def should_stop(lf, t, confidence, nv_guard):

    # 6번째 발화부터 판단
    if t < 6:
        return False

    # 최근 3회 예측
    s0, s1, s2 = (
        preds[lf][j][1]
        for j in (t-2, t-1, t)
    )

    # 빈 예측으로는 조기 중단하지 않음
    if not s2:
        return False

    # 3회 연속 동일 예측
    if not (s0 == s1 == s2):
        return False

    # 최근 3회 확률
    p0, p1, p2 = (
        probs[lf][j]
        for j in (t-2, t-1, t)
    )

    # Delta 계산
    delta = max(
        max(abs(a-b) for a,b in zip(p0,p1)),
        max(abs(a-b) for a,b in zip(p1,p2))
    )

    if delta > 0.05:
        return False

    # --------------------------------------
    # Confidence Guard
    # --------------------------------------

    if confidence is not None:
        for symptom in s2:
            i = IDX[symptom]

            if p2[i] < confidence:
                return False

    # --------------------------------------
    # 구토 / 오심 혼동 Guard
    # --------------------------------------

    if nv_guard:

        nausea = "오심" in s2
        vomit = "구토" in s2

        # 둘 중 하나만 예측한 경우
        if nausea != vomit:

            if nausea:
                other_prob = p2[IDX["구토"]]
            else:
                other_prob = p2[IDX["오심"]]

            # 다른 증상의 확률이
            # 0.5 근처면 중단을 보류
            if 0.40 <= other_prob <= 0.60:
                return False

    return True


# ==========================================
# 5. 순차 입력 시뮬레이션
# ==========================================

for name, confidence, guard in CONFIGS:

    output = out_dir / f"{name}.csv"

    halt_steps = []
    early_stops = 0

    with output.open(
        "w", encoding="utf-8", newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "label_file",
            "halt_step",
            "symptom"
        ])

        for lf, rows in preds.items():

            chosen = None
            last_observed = None

            # 1번째 발화부터 순차적으로 입력
            for t in range(1, len(rows) + 1):

                last_observed = t

                if should_stop(
                    lf, t, confidence, guard
                ):
                    chosen = t
                    early_stops += 1
                    break

            # 중단하지 못하면 실제 대화 종료
            if chosen is None:
                chosen = last_observed

            writer.writerow([
                lf,
                chosen,
                rows[chosen][0]
            ])

            halt_steps.append(chosen)

    assert len(halt_steps) == 5840

    print("\n" + "=" * 55)
    print("실험:", name)
    print("전체 통화:", len(halt_steps))
    print("평균 중단:", round(mean(halt_steps), 2))
    print("중앙값:", median(halt_steps))
    print("조기 중단 통화:", early_stops)
    print(
        "10회 이내 중단:",
        f"{sum(t <= 10 for t in halt_steps) / len(halt_steps) * 100:.2f}%"
    )
    print("저장:", output)


# ==========================================
# 6. 기존 최고 정책과 일치 여부 검증
# ==========================================

old_file = out_dir / "stable3_delta05.csv"
new_file = out_dir / "base_3_d050.csv"

if old_file.exists():

    def read_result(path):
        with path.open(
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            assert {
                "label_file",
                "halt_step",
                "symptom"
            }.issubset(reader.fieldnames or [])

            return {
                r["label_file"]: (
                    r["halt_step"],
                    normalize(r["symptom"])
                )
                for r in reader
            }

    old = read_result(old_file)
    new = read_result(new_file)

    assert old == new, (
        "기존 최고 정책과 예측 불일치! "
        "채점 전 구현을 확인해야 합니다."
    )

    print("\n[PASS] 기존 정책 예측 완전 일치")

else:
    print("\n[주의] 기존 비교 CSV 없음")

print("\n전체 실험 완료!")
