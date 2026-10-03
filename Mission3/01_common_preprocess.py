# ============================================================
# Mission 3 Common Preprocessing
#
# 목적
# 1. 여러 압축파일/폴더의 Training JSON 자동 탐색
# 2. utterances[].text만 추출
# 3. 대화 전체 text 생성
# 4. 9개 target symptom만 multi-hot encoding
# 5. Training 내부에서 고정 train/dev split 생성
# 6. CSV / JSONL / split manifest 저장
#
# 중요:
# - speaker 사용 X
# - startAt / endAt 사용 X
# - gender 사용 X
# - triage / urgency / address 등 사용 X
# - Validation 폴더의 symptom은 학습에 사용 X
# ============================================================

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# 1. 경로 설정
# ============================================================

# 여러 압축파일을 풀어놓은 최상위 폴더
DATA_ROOT = Path.home() / "DCC" / "M3_JSON"
# 결과 저장
OUT_DIR = DATA_ROOT / "Work" / "M3_Common"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Mission 3 Target
# ============================================================

TARGET_SYMPTOMS = [
    "고열",
    "구토",
    "두통",
    "복통",
    "어지러움",
    "열상",
    "오심",
    "전신쇠약",
    "호흡곤란",
]

LABEL_COLUMNS = [
    f"label_{x}"
    for x in TARGET_SYMPTOMS
]


# ============================================================
# 3. Split 설정
# ============================================================

DEV_RATIO = 0.20
RANDOM_SEED = 42


# ============================================================
# 4. 텍스트 정리
# ============================================================

def clean_text(text):
    """
    의미를 바꾸는 전처리는 하지 않는다.

    수행:
    - None 방어
    - 앞/뒤 공백 제거
    - 여러 개의 공백만 하나로 정리

    하지 않음:
    - 형태소 분석
    - stopword 제거
    - punctuation 제거
    - 띄어쓰기 교정
    - 맞춤법 교정
    - 증상 단어 치환
    """

    if not isinstance(text, str):
        return ""

    text = text.strip()

    # space / tab 반복만 정리
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text


# ============================================================
# 5. JSON 파일 탐색
# ============================================================

def find_json_files(
    root,
    split="Training",
    seoul_only=True
):
    """
    DATA_ROOT 아래를 전부 탐색해서

    .../Training/2.라벨링데이터/.../*.json

    형태의 파일을 자동으로 수집한다.

    여러 shard:
    -001
    -003
    -004
    ...

    형태로 압축이 풀려 있어도 전부 탐색 가능.
    """

    result = []

    for path in root.rglob("*.json"):

        parts = path.parts

        # Training / Validation 구분
        if split not in parts:
            continue

        # 라벨링데이터만
        if "2.라벨링데이터" not in parts:
            continue

        # Training은 서울 데이터만 사용
        if (
            seoul_only
            and split == "Training"
            and "서울" not in str(path)
        ):
            continue

        result.append(path)

    return sorted(result)


# ============================================================
# 6. 한 JSON → Mission 3 sample
# ============================================================

def parse_training_json(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # --------------------------------------------------------
    # 6-1. Transcript text만 사용
    # --------------------------------------------------------

    utterance_texts = []

    for utt in data.get(
        "utterances",
        []
    ):

        # speaker / startAt / endAt 사용하지 않음
        text = clean_text(
            utt.get("text", "")
        )

        if text:
            utterance_texts.append(
                text
            )

    # 발화 순서를 그대로 유지하면서 연결
    full_text = "\n".join(
        utterance_texts
    )

    # --------------------------------------------------------
    # 6-2. target symptom만 유지
    # --------------------------------------------------------

    raw_symptoms = data.get(
        "symptom",
        []
    )

    if not isinstance(
        raw_symptoms,
        list
    ):
        raw_symptoms = []

    raw_symptom_set = set(
        raw_symptoms
    )

    # 문제에서 지정한 9개만 사용
    target_labels = [
        symptom
        for symptom in TARGET_SYMPTOMS
        if symptom in raw_symptom_set
    ]

    # --------------------------------------------------------
    # 6-3. multi-hot encoding
    # --------------------------------------------------------

    label_vector = [
        int(
            symptom in raw_symptom_set
        )
        for symptom in TARGET_SYMPTOMS
    ]

    # --------------------------------------------------------
    # 6-4. sample id
    # --------------------------------------------------------

    # 제출 시 label file name이 필요하므로
    # 파일명을 반드시 보존
    label_file = json_path.name

    # JSON 자체의 id도 별도 보관
    call_id = data.get(
        "_id",
        json_path.stem
    )

    # --------------------------------------------------------
    # 6-5. utterance list
    #
    # Hierarchical / chunking 연구를 위해
    # '텍스트만' 리스트 형태로 보존
    # --------------------------------------------------------

    utterances_json = json.dumps(
        utterance_texts,
        ensure_ascii=False
    )

    labels_json = json.dumps(
        target_labels,
        ensure_ascii=False
    )

    row = {
        "call_id": call_id,
        "sample_key": json_path.stem,
        "label_file": label_file,

        "text": full_text,

        # text 이외 annotation이 아니라
        # utterance text 배열 자체
        "utterances_json": utterances_json,

        "labels_json": labels_json,

        # 분석용
        "num_utterances": len(
            utterance_texts
        ),
        "num_chars": len(
            full_text
        ),
        "num_labels": len(
            target_labels
        ),
    }

    # label_고열 ~ label_호흡곤란
    for symptom, value in zip(
        TARGET_SYMPTOMS,
        label_vector
    ):

        row[
            f"label_{symptom}"
        ] = value

    return row


# ============================================================
# 7. 전체 Training 데이터 생성
# ============================================================

print("=" * 60)
print("Mission 3 Training JSON 검색")
print("=" * 60)

train_json_files = find_json_files(
    DATA_ROOT,
    split="Training",
    seoul_only=True
)

print(
    "발견된 Training JSON:",
    len(train_json_files)
)


# ============================================================
# 8. 중복 filename 확인
# ============================================================

file_counter = Counter(
    p.name
    for p in train_json_files
)

duplicates = [
    name
    for name, count
    in file_counter.items()
    if count > 1
]

if duplicates:

    print(
        "\n[ERROR] 동일 JSON 파일명 중복:",
        len(duplicates)
    )

    for name in duplicates[:20]:
        print(name)

    raise RuntimeError(
        "동일 label filename이 여러 곳에서 발견되었습니다."
    )


# ============================================================
# 9. JSON Parsing
# ============================================================

rows = []

errors = []

for i, json_path in enumerate(
    train_json_files,
    start=1
):

    try:

        row = parse_training_json(
            json_path
        )

        rows.append(row)

    except Exception as e:

        errors.append(
            (
                str(json_path),
                str(e)
            )
        )

    if (
        i % 1000 == 0
        or i == len(
            train_json_files
        )
    ):

        print(
            f"\rParsing: "
            f"{i}/"
            f"{len(train_json_files)}",
            end=""
        )

print()


# ============================================================
# 10. DataFrame
# ============================================================

df = pd.DataFrame(rows)

print("\n정상 sample:", len(df))
print("Parsing error:", len(errors))


# ============================================================
# 11. 데이터 무결성 검사
# ============================================================

print("\n")
print("=" * 60)
print("데이터 검사")
print("=" * 60)


# sample key 중복
duplicate_samples = (
    df["sample_key"]
    .duplicated()
    .sum()
)

print(
    "중복 sample_key:",
    duplicate_samples
)


# 빈 transcript
empty_text_count = (
    df["text"]
    .fillna("")
    .str.len()
    .eq(0)
    .sum()
)

print(
    "빈 transcript:",
    empty_text_count
)


# label이 하나도 없는 sample
zero_label_count = (
    df[LABEL_COLUMNS]
    .sum(axis=1)
    .eq(0)
    .sum()
)

print(
    "Target symptom 0개 sample:",
    zero_label_count
)


# ============================================================
# 12. Label 분포
# ============================================================

print("\n")
print("=" * 60)
print("전체 Label 분포")
print("=" * 60)

label_stats = []

for symptom in TARGET_SYMPTOMS:

    col = f"label_{symptom}"

    count = int(
        df[col].sum()
    )

    ratio = (
        count / len(df)
        if len(df)
        else 0
    )

    label_stats.append({
        "symptom": symptom,
        "count": count,
        "ratio": ratio
    })

    print(
        f"{symptom:6s}: "
        f"{count:6d} "
        f"({ratio:.4%})"
    )


label_stats_df = pd.DataFrame(
    label_stats
)


# ============================================================
# 13. Train / Dev 고정 Split
# ============================================================
#
# 가장 좋은 방법:
# iterative multilabel stratification
#
# 패키지가 있으면 자동 사용.
#
# pip install iterative-stratification
#
# 없으면 deterministic random split으로 fallback.
# ============================================================

indices = np.arange(
    len(df)
)

Y = df[
    LABEL_COLUMNS
].values


try:

    from iterstrat.ml_stratifiers import (
        MultilabelStratifiedShuffleSplit
    )

    splitter = (
        MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=DEV_RATIO,
            random_state=RANDOM_SEED
        )
    )

    train_idx, dev_idx = next(
        splitter.split(
            indices,
            Y
        )
    )

    split_method = (
        "MultilabelStratifiedShuffleSplit"
    )

except ImportError:

    print(
        "\n[WARNING]"
        "\niterative-stratification이 설치되어 있지 않습니다."
        "\n일반 random split을 사용합니다."
        "\n"
        "\n권장 설치:"
        "\npip install iterative-stratification"
        "\n"
    )

    train_idx, dev_idx = (
        train_test_split(
            indices,
            test_size=DEV_RATIO,
            random_state=RANDOM_SEED,
            shuffle=True
        )
    )

    split_method = (
        "RandomSplit"
    )


train_df = (
    df.iloc[train_idx]
    .copy()
    .reset_index(drop=True)
)

dev_df = (
    df.iloc[dev_idx]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# 14. Split column 추가
# ============================================================

df["split"] = ""

df.loc[
    train_idx,
    "split"
] = "train"

df.loc[
    dev_idx,
    "split"
] = "dev"


# ============================================================
# 15. Split 상태 출력
# ============================================================

print("\n")
print("=" * 60)
print("Train / Dev Split")
print("=" * 60)

print(
    "Split method:",
    split_method
)

print(
    "Train:",
    len(train_df)
)

print(
    "Dev:",
    len(dev_df)
)


print(
    "\n===== Train label ====="
)

for symptom in TARGET_SYMPTOMS:

    col = f"label_{symptom}"

    print(
        f"{symptom:6s}: "
        f"{int(train_df[col].sum())}"
    )


print(
    "\n===== Dev label ====="
)

for symptom in TARGET_SYMPTOMS:

    col = f"label_{symptom}"

    print(
        f"{symptom:6s}: "
        f"{int(dev_df[col].sum())}"
    )


# ============================================================
# 16. CSV 저장
# ============================================================

ALL_PATH = (
    OUT_DIR /
    "mission3_all.csv"
)

TRAIN_PATH = (
    OUT_DIR /
    "mission3_train.csv"
)

DEV_PATH = (
    OUT_DIR /
    "mission3_dev.csv"
)

SPLIT_PATH = (
    OUT_DIR /
    "split_manifest.csv"
)

LABEL_STATS_PATH = (
    OUT_DIR /
    "label_stats.csv"
)


df.to_csv(
    ALL_PATH,
    index=False,
    encoding="utf-8-sig"
)

train_df.to_csv(
    TRAIN_PATH,
    index=False,
    encoding="utf-8-sig"
)

dev_df.to_csv(
    DEV_PATH,
    index=False,
    encoding="utf-8-sig"
)


df[
    [
        "call_id",
        "sample_key",
        "label_file",
        "split"
    ]
].to_csv(
    SPLIT_PATH,
    index=False,
    encoding="utf-8-sig"
)


label_stats_df.to_csv(
    LABEL_STATS_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. JSONL 저장
#
# Hierarchical / Transformer 등에서 편하게 사용
# ============================================================

JSONL_PATH = (
    OUT_DIR /
    "mission3_all.jsonl"
)

with open(
    JSONL_PATH,
    "w",
    encoding="utf-8"
) as f:

    for _, row in df.iterrows():

        obj = {
            "call_id":
                row["call_id"],

            "sample_key":
                row["sample_key"],

            "label_file":
                row["label_file"],

            "text":
                row["text"],

            "utterances":
                json.loads(
                    row["utterances_json"]
                ),

            "labels":
                json.loads(
                    row["labels_json"]
                ),

            "label_vector": [
                int(
                    row[
                        f"label_{symptom}"
                    ]
                )
                for symptom
                in TARGET_SYMPTOMS
            ],

            "split":
                row["split"]
        }

        f.write(
            json.dumps(
                obj,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# 18. Label Map 저장
# ============================================================

LABEL_MAP_PATH = (
    OUT_DIR /
    "label_map.json"
)

label_map = {
    "labels": TARGET_SYMPTOMS,
    "label2id": {
        label: i
        for i, label
        in enumerate(
            TARGET_SYMPTOMS
        )
    },
    "id2label": {
        str(i): label
        for i, label
        in enumerate(
            TARGET_SYMPTOMS
        )
    }
}

with open(
    LABEL_MAP_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        label_map,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 19. Validation 입력 데이터 생성
#
# 중요:
# Validation의 symptom은 절대로 가져오지 않는다.
# 오직 text만 추출한다.
# ============================================================

def parse_validation_json(
    json_path
):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    texts = []

    for utt in data.get(
        "utterances",
        []
    ):

        text = clean_text(
            utt.get(
                "text",
                ""
            )
        )

        if text:
            texts.append(text)

    return {
        "call_id":
            data.get(
                "_id",
                json_path.stem
            ),

        "sample_key":
            json_path.stem,

        "label_file":
            json_path.name,

        "text":
            "\n".join(texts),

        "utterances_json":
            json.dumps(
                texts,
                ensure_ascii=False
            )
    }


validation_json_files = (
    find_json_files(
        DATA_ROOT,
        split="Validation",
        seoul_only=False
    )
)

if validation_json_files:

    validation_rows = []

    for json_path in (
        validation_json_files
    ):

        try:

            validation_rows.append(
                parse_validation_json(
                    json_path
                )
            )

        except Exception as e:

            print(
                "[Validation ERROR]",
                json_path,
                e
            )

    validation_df = pd.DataFrame(
        validation_rows
    )

    VALIDATION_PATH = (
        OUT_DIR /
        "official_validation_input.csv"
    )

    validation_df.to_csv(
        VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig"
    )

else:

    VALIDATION_PATH = None


# ============================================================
# 20. 완료
# ============================================================

print("\n")
print("=" * 60)
print("Mission 3 공통 전처리 완료")
print("=" * 60)

print(
    "전체 Training:",
    len(df)
)

print(
    "Train:",
    len(train_df)
)

print(
    "Dev:",
    len(dev_df)
)

print(
    "빈 transcript:",
    empty_text_count
)

print(
    "Target 0-label sample:",
    zero_label_count
)

print("\n저장 위치:")
print(OUT_DIR)

print("\n생성 파일:")
print(" -", ALL_PATH.name)
print(" -", TRAIN_PATH.name)
print(" -", DEV_PATH.name)
print(" -", SPLIT_PATH.name)
print(" -", LABEL_STATS_PATH.name)
print(" -", JSONL_PATH.name)
print(" -", LABEL_MAP_PATH.name)

if VALIDATION_PATH:
    print(
        " -",
        VALIDATION_PATH.name
    )

if errors:

    print(
        "\nParsing error 예시:"
    )

    for path, error in errors[:10]:

        print(
            path,
            "→",
            error
        )
