"""
05b_preprocess_provided_validation.py

역할: 02_common_preprocess.py와 거의 동일한 신고자(caller, speaker==1) segment
      추출 로직을 그대로 사용하되, 아래 두 가지만 다름:

      1. 고정 split CSV(train_calls.csv/val_calls.csv)에 의존하지 않음.
         -> Data\\Validation은 그 자체로 하나의 held-out 세트이고,
            대회 측이 제공한 별도 파티션이므로 자체 gender 정답(JSON 내부의
            "gender" 필드)을 그대로 신뢰해서 사용.
      2. 결과물을 완전히 별도 폴더(Work\\ProvidedValidation\\Common)에 저장.
         -> 기존 Work\\Common (Training 파이프라인 결과물)은 절대 건드리지 않음.

      추가 안전장치: 혹시 모를 leakage 확인을 위해, Data\\Validation의 call_id들이
      팀 공통 고정 Split(train_calls.csv/val_calls.csv)의 call_id와 겹치는지도 확인.
      (단, 이건 call_id 기준 검사이고, 화자 단위 중복까지 보장하지는 않음)
"""

import json
import csv
import shutil
from pathlib import Path
from collections import Counter

import soundfile as sf


ROOT_DIR = Path(__file__).resolve().parent.parent  # Code\ 의 부모 = DCC 루트

WORK_DIR = ROOT_DIR / "Work"
SHARED_DIR = ROOT_DIR / "Shared"

PV_WAV_DIR = WORK_DIR / "ProvidedValidation" / "wav"
PV_JSON_DIR = WORK_DIR / "ProvidedValidation" / "json"

PV_COMMON_DIR = WORK_DIR / "ProvidedValidation" / "Common"
PV_SEGMENT_DIR = PV_COMMON_DIR / "caller_segments"
PV_METADATA_PATH = PV_COMMON_DIR / "metadata.csv"

# leakage 교차확인용 (읽기만 함, 절대 덮어쓰지 않음)
FIXED_TRAIN_CALLS_PATH = SHARED_DIR / "train_calls.csv"
FIXED_VAL_CALLS_PATH = SHARED_DIR / "val_calls.csv"

CALLER_SPEAKER = 1
VALID_GENDERS = {"M", "F"}
RESET_OUTPUT = True


print("DCC 루트로 인식된 경로:", ROOT_DIR)

for path in [PV_WAV_DIR, PV_JSON_DIR]:
    if not path.exists():
        raise FileNotFoundError(
            f"필수 경로가 없습니다:\n{path}\n"
            f"05a_copy_provided_validation.py를 먼저 실행하세요."
        )

if RESET_OUTPUT and PV_COMMON_DIR.exists():
    print("기존 ProvidedValidation\\Common 결과 삭제:", PV_COMMON_DIR)
    shutil.rmtree(PV_COMMON_DIR)

PV_SEGMENT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 파일 인덱스 (02번과 동일 로직)
# =========================================================

def build_index(root, ext):
    root = Path(root)
    index = {}
    duplicates = []

    for path in root.rglob(f"*{ext}"):
        stem = path.stem
        if stem in index:
            duplicates.append((stem, index[stem], path))
        else:
            index[stem] = path

    if duplicates:
        print("\n[ERROR] 중복 stem 발견")
        for stem, first, second in duplicates[:10]:
            print(f"\nstem: {stem}\n1: {first}\n2: {second}")
        raise RuntimeError(f"{ext} 중복 stem {len(duplicates)}개 발견")

    return index


print("파일 인덱스 생성 중...")
wav_index = build_index(PV_WAV_DIR, ".wav")
json_index = build_index(PV_JSON_DIR, ".json")
print("WAV:", len(wav_index))
print("JSON:", len(json_index))


# =========================================================
# WAV / JSON 매칭
# =========================================================

wav_stems = set(wav_index.keys())
json_stems = set(json_index.keys())
matched_stems = wav_stems & json_stems
missing_wav = json_stems - wav_stems
missing_json = wav_stems - json_stems

print()
print("=" * 60)
print("WAV / JSON 매칭")
print("=" * 60)
print("정상 매칭:", len(matched_stems))
print("WAV가 없는 JSON:", len(missing_wav))
print("JSON이 없는 WAV:", len(missing_json))

if missing_wav or missing_json:
    raise RuntimeError(
        "WAV/JSON 매칭이 완전하지 않습니다. Data\\Validation 다운로드 상태를 확인하세요."
    )


# =========================================================
# (안전장치) 팀 공통 고정 Split과 call_id 중복(leakage) 확인
# =========================================================

def load_call_ids(csv_path):
    ids = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(str(row["call_id"]).strip())
    return ids


if FIXED_TRAIN_CALLS_PATH.exists() and FIXED_VAL_CALLS_PATH.exists():
    fixed_train_ids = load_call_ids(FIXED_TRAIN_CALLS_PATH)
    fixed_val_ids = load_call_ids(FIXED_VAL_CALLS_PATH)
    fixed_all_ids = fixed_train_ids | fixed_val_ids

    overlap = matched_stems & fixed_all_ids

    print()
    print("=" * 60)
    print("Training 파티션 고정 Split과 call_id 중복(leakage) 확인")
    print("=" * 60)
    print("겹치는 call_id 수:", len(overlap))

    if overlap:
        print("\n[ERROR] 겹치는 call_id 예시:")
        for call_id in sorted(overlap)[:10]:
            print(" ", call_id)
        raise RuntimeError(
            "\nData\\Validation의 call_id가 Training 파티션 고정 Split과 겹칩니다.\n"
            "같은 call이 두 파티션에 동시에 존재하면 안 되므로 데이터를 다시 확인하세요."
        )
    else:
        print("✅ 겹치는 call_id 없음 (call_id 기준으로는 독립적인 파티션 확인됨)")
else:
    print("\n[알림] Shared\\train_calls.csv / val_calls.csv를 찾을 수 없어 leakage 교차확인을 건너뜁니다.")


# =========================================================
# 신고자(caller) segment 추출 (02번과 동일 로직, gender는 JSON 자체 값 사용)
# =========================================================

def extract_caller_segments(json_path, wav_path, stem):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_gender = data.get("gender")

    if json_gender not in VALID_GENDERS:
        raise RuntimeError(f"잘못된 gender: {stem} -> {json_gender}")

    utterances = [
        utt for utt in data.get("utterances", [])
        if utt.get("speaker") == CALLER_SPEAKER
    ]
    utterances.sort(key=lambda x: x.get("startAt", 0))

    y, sr = sf.read(wav_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    audio_length = len(y)
    rows = []
    saved_segment_id = 0

    for utt in utterances:
        start_ms = utt.get("startAt")
        end_ms = utt.get("endAt")

        if start_ms is None or end_ms is None:
            continue
        if end_ms <= start_ms:
            continue

        start_sample = round(start_ms / 1000.0 * sr)
        end_sample = round(end_ms / 1000.0 * sr)
        start_sample = max(0, min(start_sample, audio_length))
        end_sample = max(0, min(end_sample, audio_length))

        if end_sample <= start_sample:
            continue

        clip = y[start_sample:end_sample]

        output_name = f"{stem}_seg{saved_segment_id:03d}.wav"
        output_path = PV_SEGMENT_DIR / output_name
        sf.write(output_path, clip, sr)

        duration_ms = len(clip) / sr * 1000.0

        rows.append({
            "call_id": stem,
            "segment_id": saved_segment_id,
            "segment_file": output_name,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": round(duration_ms, 3),
            "sample_rate": sr,
            "gender": json_gender,
            "split": "provided_validation",
        })

        saved_segment_id += 1

    return rows


all_metadata = []
failed_calls = []
gender_count = Counter()

all_call_ids = sorted(matched_stems)

print()
print("=" * 60)
print("신고자 Segment 추출 시작")
print("=" * 60)
print("처리 대상:", len(all_call_ids))

for i, stem in enumerate(all_call_ids, start=1):
    try:
        rows = extract_caller_segments(json_index[stem], wav_index[stem], stem)
    except Exception as e:
        print()
        print("[ERROR]", stem)
        print(e)
        failed_calls.append(stem)
        continue

    if not rows:
        print()
        print("[ERROR] 유효한 신고자 segment 없음:", stem)
        failed_calls.append(stem)
        continue

    all_metadata.extend(rows)
    gender_count[rows[0]["gender"]] += 1

    if i % 100 == 0 or i == len(all_call_ids):
        print(f"\r처리 중: {i}/{len(all_call_ids)}", end="")

print()

if failed_calls:
    print()
    print("실패 통화 수:", len(failed_calls))
    print("실패 예시:")
    for call_id in failed_calls[:10]:
        print(" ", call_id)
    raise RuntimeError("\n일부 통화 처리에 실패했습니다. 위 에러를 확인하세요.")


metadata_fields = [
    "call_id", "segment_id", "segment_file", "start_ms", "end_ms",
    "duration_ms", "sample_rate", "gender", "split",
]

with open(PV_METADATA_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=metadata_fields)
    writer.writeheader()
    writer.writerows(all_metadata)


print()
print("=" * 60)
print("Provided-Validation 전처리 완료")
print("=" * 60)
print("전체 WAV:", len(wav_index))
print("전체 JSON:", len(json_index))
print("처리된 통화:", len(all_call_ids))
print("생성된 신고자 segment:", len(all_metadata))
print()
print("성별 분포 (통화 기준):")
for gender, count in sorted(gender_count.items()):
    print(f"  {gender}: {count}")

print()
print("Caller segments:", PV_SEGMENT_DIR)
print("Metadata:", PV_METADATA_PATH)
