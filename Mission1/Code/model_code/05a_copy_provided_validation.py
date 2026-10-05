"""
05a_copy_provided_validation.py

역할: 01_extract_training.py와 완전히 동일한 로직으로,
      Data\\Validation (대회 제공 별도 검증 파티션, 3,640개)의
      wav/json을 Work\\ProvidedValidation\\wav, \\json 으로 모음.

주의: Data\\Training 쪽 (Work\\Training, Work\\Common)은 전혀 건드리지 않음.
      완전히 분리된 새 폴더에서만 작업.
"""

from pathlib import Path
import shutil
import zlib


BASE_DIR = Path(__file__).resolve().parent.parent  # Code\ 의 부모 = DCC 루트

DATA_DIR = BASE_DIR / "Data"

# 대회 제공 Validation 파티션 (하위 구조는 몰라도 됨, 재귀 탐색함)
VALIDATION_ROOT = DATA_DIR / "Validation"

WORK_DIR = BASE_DIR / "Work"
PV_ROOT = WORK_DIR / "ProvidedValidation"

WAV_OUT_DIR = PV_ROOT / "wav"
JSON_OUT_DIR = PV_ROOT / "json"

WAV_OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TOTAL = 3640  # 대회 규정 문서 기준 Validation 데이터셋 크기


print("DCC 루트로 인식된 경로:", BASE_DIR)

if not VALIDATION_ROOT.exists():
    raise FileNotFoundError(
        f"경로 없음: {VALIDATION_ROOT}\n"
        f"이 스크립트가 'DCC\\Code\\' 폴더 안에 있는지 확인하세요.\n"
        f"(현재 스크립트 위치: {Path(__file__).resolve()})"
    )

print("사용하는 원본 폴더:", VALIDATION_ROOT)
print("(Data\\Training 폴더는 이 스크립트에서 사용하지 않습니다)")


def calculate_crc32(file_path):
    """파일 내용을 1MB씩 읽어 CRC32 체크섬을 계산한다."""
    crc = 0
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def copy_file(src_path, out_dir):
    """파일명이 같은 게 이미 있으면 내용을 비교해서, 같으면 건너뛰고
    다르면 에러를 내서 데이터 불일치를 바로 알아챌 수 있게 한다."""
    target = out_dir / src_path.name

    if target.exists():
        if calculate_crc32(src_path) == calculate_crc32(target):
            return "duplicate_same"
        raise RuntimeError(
            f"동일 파일명인데 내용이 다릅니다: {src_path.name}\n"
            f"1: {target}\n2: {src_path}"
        )

    shutil.copy2(src_path, target)
    return "saved"


def collect(root, ext, out_dir):
    """root 아래를 재귀 탐색해서 확장자가 ext인 파일을 전부 out_dir로 복사."""
    saved = 0
    duplicate = 0

    files = sorted(root.rglob(f"*{ext}"))

    for i, path in enumerate(files, start=1):
        if not path.is_file():
            continue

        result = copy_file(path, out_dir)
        if result == "saved":
            saved += 1
        else:
            duplicate += 1

        if i % 500 == 0 or i == len(files):
            print(f"\r  {ext} 처리 중: {i}/{len(files)}", end="")

    print()
    return saved, duplicate, len(files)


print("\nWAV 수집 중...")
wav_saved, wav_dup, wav_found = collect(VALIDATION_ROOT, ".wav", WAV_OUT_DIR)

print("\nJSON 수집 중...")
json_saved, json_dup, json_found = collect(VALIDATION_ROOT, ".json", JSON_OUT_DIR)


actual_wav = list(WAV_OUT_DIR.glob("*.wav"))
actual_json = list(JSON_OUT_DIR.glob("*.json"))

print()
print("=" * 60)
print("로컬 Provided-Validation 준비 완료")
print("=" * 60)

print("발견된 WAV:", wav_found, " (신규 저장:", wav_saved, "/ 중복:", wav_dup, ")")
print("발견된 JSON:", json_found, " (신규 저장:", json_saved, "/ 중복:", json_dup, ")")

print()
print("현재 WAV 실제 개수:", len(actual_wav))
print("현재 JSON 실제 개수:", len(actual_json))

print()
print(f"기대값 (대회 규정 문서 기준 Validation 데이터셋): {EXPECTED_TOTAL:,}")

if len(actual_wav) != EXPECTED_TOTAL or len(actual_json) != EXPECTED_TOTAL:
    print()
    print(f"⚠ 경고: 개수가 {EXPECTED_TOTAL:,}과 다릅니다. Data\\Validation 다운로드/압축 해제 상태를 확인하세요.")
else:
    print(f"✅ 정상: 규정 문서 기준({EXPECTED_TOTAL:,}개)과 일치합니다.")

print()
print("WAV 위치:", WAV_OUT_DIR)
print("JSON 위치:", JSON_OUT_DIR)
