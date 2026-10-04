"""
02_common_preprocess.py

역할: Work\\Training\\wav, Work\\Training\\json(01_extract_training.py가 모아둔 것)에서
      팀 공통 고정 split(Shared\\train_calls.csv, val_calls.csv)에 속한 통화만 골라,
      신고자(speaker=1) 발화 구간을 crop해서 Work\\Common\\caller_segments에 저장하고
      metadata.csv를 만든다.

      crop 로직 자체는 gender를 전혀 참조하지 않는다. gender는 JSON에 적힌 값과
      팀 공통 CSV의 값이 서로 일치하는지 교차검증하는 용도로만 쓰인다.

검증: 고정 split 자체의 중복, WAV/JSON 매칭, 로컬 데이터와 고정 split의 일치 여부,
      최종 metadata의 train/val leakage까지 단계마다 확인해서 하나라도 어긋나면
      중단한다.

전제: 01_extract_training.py를 먼저 실행해서 Work\\Training\\wav, \\json이 채워져
      있어야 하고, Shared\\train_calls.csv, val_calls.csv(팀 공통 고정 split)가
      있어야 한다.

출력: Work\\Common\\caller_segments\\*.wav, Work\\Common\\metadata.csv,
      Work\\Common\\train_calls.csv, val_calls.csv (고정 split 복사본)
"""

import json
import csv
import shutil
from pathlib import Path
from collections import Counter

import soundfile as sf


# ============================================================
# 경로 설정
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent  # Code\ 의 부모 = DCC 루트
WORK_DIR = ROOT_DIR / "Work"
SHARED_DIR = ROOT_DIR / "Shared"

TRAIN_WAV_DIR = WORK_DIR / "Training" / "wav"
TRAIN_JSON_DIR = WORK_DIR / "Training" / "json"

# 팀 공통 고정 split
FIXED_TRAIN_CALLS_PATH = SHARED_DIR / "train_calls.csv"
FIXED_VAL_CALLS_PATH = SHARED_DIR / "val_calls.csv"

# 출력 위치
COMMON_DIR = WORK_DIR / "Common"
SEGMENT_DIR = COMMON_DIR / "caller_segments"
METADATA_PATH = COMMON_DIR / "metadata.csv"
OUTPUT_TRAIN_CALLS_PATH = COMMON_DIR / "train_calls.csv"  # 결과 폴더에도 보관
OUTPUT_VAL_CALLS_PATH = COMMON_DIR / "val_calls.csv"

CALLER_SPEAKER = 1
VALID_GENDERS = {"M", "F"}
RESET_OUTPUT = True  # 기존 Work\Common 결과를 지우고 새로 생성


# ============================================================
# 필수 파일/폴더 확인
# ============================================================

print("DCC 루트로 인식된 경로:", ROOT_DIR)

required_paths = [TRAIN_WAV_DIR, TRAIN_JSON_DIR, FIXED_TRAIN_CALLS_PATH, FIXED_VAL_CALLS_PATH]
for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"필수 경로가 없습니다:\n{path}\n"
            f"이 스크립트가 'DCC\\Code\\' 폴더 안에 있는지,\n"
            f"01_extract_training.py를 먼저 실행했는지 확인하세요."
        )

if RESET_OUTPUT and COMMON_DIR.exists():
    print("기존 Common 결과 삭제:", COMMON_DIR)
    shutil.rmtree(COMMON_DIR)

SEGMENT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 파일 인덱스 생성
# ============================================================

def build_index(root, ext):
    """root 아래를 재귀 탐색해서 {stem: path} 인덱스를 만든다. stem이 겹치면 에러."""
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
wav_index = build_index(TRAIN_WAV_DIR, ".wav")
json_index = build_index(TRAIN_JSON_DIR, ".json")
print("WAV:", len(wav_index))
print("JSON:", len(json_index))


# ============================================================
# 고정 Split CSV 로드
# ============================================================

def load_fixed_split(csv_path, expected_split):
    """call_id -> {gender, split} 딕셔너리로 로드. 형식/중복/split값 불일치는 에러."""
    result = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"call_id", "gender"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise RuntimeError(f"{csv_path.name}에 필수 컬럼이 없습니다.\n필요: {required_columns}")

        for row in reader:
            call_id = str(row["call_id"]).strip()
            gender = str(row["gender"]).strip()

            if not call_id:
                raise RuntimeError(f"{csv_path.name}에 빈 call_id가 있습니다.")
            if gender not in VALID_GENDERS:
                raise RuntimeError(f"잘못된 gender: {call_id} -> {gender}")
            if call_id in result:
                raise RuntimeError(f"중복 call_id 발견: {call_id}")

            # split 컬럼이 있으면 값도 확인
            if "split" in row:
                split_value = str(row["split"]).strip()
                if split_value and split_value != expected_split:
                    raise RuntimeError(f"{csv_path.name}의 split 값 오류:\n{call_id} -> {split_value}")

            result[call_id] = {"gender": gender, "split": expected_split}

    return result


print()
print("=" * 60)
print("팀 공통 고정 Split 로드")
print("=" * 60)

fixed_train = load_fixed_split(FIXED_TRAIN_CALLS_PATH, "train")
fixed_val = load_fixed_split(FIXED_VAL_CALLS_PATH, "val")

train_ids = set(fixed_train.keys())
val_ids = set(fixed_val.keys())

print("고정 Train:", len(train_ids))
print("고정 Validation:", len(val_ids))


# ============================================================
# Split 자체 Leakage 검사
# ============================================================

split_overlap = train_ids & val_ids
if split_overlap:
    raise RuntimeError(f"공유받은 Train / Validation에 중복 call_id가 있습니다: {len(split_overlap)}개")

print("고정 Split overlap:", len(split_overlap))


# ============================================================
# WAV / JSON 매칭 검사
# ============================================================

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


# ============================================================
# 로컬 데이터와 고정 Split 일치 검사
# ============================================================

fixed_all_ids = train_ids | val_ids
missing_from_local = fixed_all_ids - matched_stems  # 팀원 PC에 없는 공통 call
extra_local_calls = matched_stems - fixed_all_ids    # 팀원 PC에만 있는 로컬 call

print()
print("=" * 60)
print("고정 Split / 로컬 데이터 비교")
print("=" * 60)
print("고정 Split 전체:", len(fixed_all_ids))
print("로컬 정상 매칭:", len(matched_stems))
print("로컬에 없는 공통 call:", len(missing_from_local))
print("공통 Split에 없는 로컬 call:", len(extra_local_calls))

if missing_from_local:
    print("\n[ERROR] 팀원 PC에 없는 공통 call 예시:")
    for call_id in sorted(missing_from_local)[:10]:
        print(" ", call_id)

if extra_local_calls:
    print("\n[ERROR] 공통 Split에 없는 로컬 call 예시:")
    for call_id in sorted(extra_local_calls)[:10]:
        print(" ", call_id)

if missing_from_local or extra_local_calls:
    raise RuntimeError(
        "\n팀 공통 데이터와 현재 PC 데이터가 일치하지 않습니다.\n"
        "01_extract_training.py 결과와 Data\\Training 폴더를 확인하세요."
    )


# ============================================================
# Gender 정답 Dictionary
# ============================================================

fixed_gender = {}
for call_id, info in fixed_train.items():
    fixed_gender[call_id] = info["gender"]
for call_id, info in fixed_val.items():
    fixed_gender[call_id] = info["gender"]


# ============================================================
# 신고자 Segment 추출
# ============================================================

def extract_caller_segments(json_path, wav_path, stem, split, expected_gender):
    """JSON의 speaker==1 구간만 crop해서 Work\\Common\\caller_segments에 저장하고
    metadata 행 목록을 반환한다. gender는 교차검증(JSON vs 공통 CSV)에만 쓰이고
    crop 자체에는 전혀 관여하지 않는다."""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_gender = data.get("gender")
    if json_gender != expected_gender:
        raise RuntimeError(f"Gender 불일치:\n{stem}\n공통 CSV: {expected_gender}\nJSON: {json_gender}")
    if json_gender not in VALID_GENDERS:
        raise RuntimeError(f"잘못된 gender: {stem} -> {json_gender}")

    utterances = [utt for utt in data.get("utterances", []) if utt.get("speaker") == CALLER_SPEAKER]
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
        output_path = SEGMENT_DIR / output_name
        sf.write(output_path, clip, sr)  # 원본 sample rate 그대로 저장

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
            "split": split,
        })

        saved_segment_id += 1

    return rows


# ============================================================
# 전체 통화 처리
# ============================================================

all_metadata = []
failed_calls = []
gender_count = Counter()
segment_count_per_call = Counter()

all_call_ids = sorted(fixed_all_ids)  # 공유 Split에 존재하는 29,200개 call만 처리

print()
print("=" * 60)
print("신고자 Segment 추출 시작")
print("=" * 60)
print("처리 대상:", len(all_call_ids))

for i, stem in enumerate(all_call_ids, start=1):
    if stem in train_ids:
        split = "train"
    elif stem in val_ids:
        split = "val"
    else:
        raise RuntimeError(f"Split을 찾을 수 없음: {stem}")

    expected_gender = fixed_gender[stem]

    try:
        rows = extract_caller_segments(
            json_path=json_index[stem], wav_path=wav_index[stem],
            stem=stem, split=split, expected_gender=expected_gender,
        )
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
    gender_count[expected_gender] += 1
    segment_count_per_call[len(rows)] += 1

    if i % 100 == 0 or i == len(all_call_ids):
        print(f"\r처리 중: {i}/{len(all_call_ids)}", end="")

print()


# ============================================================
# 실패 통화가 있으면 중단
# ============================================================

if failed_calls:
    print()
    print("실패 통화 수:", len(failed_calls))
    print("실패 예시:")
    for call_id in failed_calls[:10]:
        print(" ", call_id)
    raise RuntimeError("\n공통 기준과 동일한 결과를 만들 수 없으므로 중단합니다.")


# ============================================================
# metadata.csv 저장
# ============================================================

metadata_fields = [
    "call_id", "segment_id", "segment_file", "start_ms", "end_ms",
    "duration_ms", "sample_rate", "gender", "split",
]

with open(METADATA_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=metadata_fields)
    writer.writeheader()
    writer.writerows(all_metadata)


# ============================================================
# 고정 Split CSV 복사 (결과 폴더에도 보관)
# ============================================================

shutil.copy2(FIXED_TRAIN_CALLS_PATH, OUTPUT_TRAIN_CALLS_PATH)
shutil.copy2(FIXED_VAL_CALLS_PATH, OUTPUT_VAL_CALLS_PATH)


# ============================================================
# 최종 Metadata Leakage 검사
# ============================================================

metadata_train_ids = {row["call_id"] for row in all_metadata if row["split"] == "train"}
metadata_val_ids = {row["call_id"] for row in all_metadata if row["split"] == "val"}
metadata_overlap = metadata_train_ids & metadata_val_ids

if metadata_overlap:
    raise RuntimeError("최종 metadata에 Train/Val leakage가 있습니다.")


# ============================================================
# 원래 고정 Split과 최종 결과가 같은지 확인
# ============================================================

if metadata_train_ids != train_ids:
    raise RuntimeError("최종 Train call_id가 공유 Train Split과 다릅니다.")
if metadata_val_ids != val_ids:
    raise RuntimeError("최종 Validation call_id가 공유 Validation Split과 다릅니다.")


# ============================================================
# 성별 분포
# ============================================================

train_gender = Counter(fixed_gender[call_id] for call_id in train_ids)
val_gender = Counter(fixed_gender[call_id] for call_id in val_ids)


# ============================================================
# 최종 결과 출력
# ============================================================

print("\n")
print("=" * 60)
print("공통 전처리 완료")
print("=" * 60)
print("전체 WAV:", len(wav_index))
print("전체 JSON:", len(json_index))
print("고정 Split 전체:", len(fixed_all_ids))
print("생성된 신고자 segment:", len(all_metadata))

print()
print("===== Train =====")
print("통화:", len(train_ids))
for gender, count in sorted(train_gender.items()):
    print(f"{gender}: {count}")

print()
print("===== Validation =====")
print("통화:", len(val_ids))
for gender, count in sorted(val_gender.items()):
    print(f"{gender}: {count}")

print()
print("===== Leakage =====")
print("Train / Val 중복 call_id:", len(metadata_overlap))

print()
print("===== Segment 개수 =====")
for segment_count, call_count in sorted(segment_count_per_call.items()):
    print(f"{segment_count}개 segment → {call_count}개 통화")

print()
print("=" * 60)
print("팀 공통 Split과 완전히 동일합니다.")
print("=" * 60)

print()
print("Caller segments:")
print(SEGMENT_DIR)
print()
print("Metadata:")
print(METADATA_PATH)
print()
print("Train calls:")
print(OUTPUT_TRAIN_CALLS_PATH)
print()
print("Validation calls:")
print(OUTPUT_VAL_CALLS_PATH)
