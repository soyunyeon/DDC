"""
01_extract_training.py

역할: AI-Hub에서 받은 Data\\Training 원본(원천데이터 wav + 라벨링데이터 json)을
      재귀 탐색해서 Work\\Training\\wav, Work\\Training\\json으로 한 곳에 모은다.

      하위 폴더 구조(예: "1.원천데이터(wav)", "2.라벨링데이터(json)")는 몰라도
      되고, 확장자만 보고 전부 찾아서 복사한다. 동일 파일명이 이미 있으면
      CRC32로 내용을 비교해서, 같으면 건너뛰고 다르면 에러를 내서 바로 알 수
      있게 한다.

전제: 이 파일이 "DCC\\Code\\" 폴더 안에 있어야 한다. 스크립트 위치를 기준으로
      DCC 루트를 자동으로 찾으므로, DCC 폴더 자체의 위치(D:\\DCC, Downloads\\DCC
      등)는 상관없다.

출력: Work\\Training\\wav\\*.wav, Work\\Training\\json\\*.json
      (Data\\Validation은 이 스크립트에서 전혀 건드리지 않는다)
"""

from pathlib import Path
import shutil
import zlib


# ============================================================
# 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # Code\ 의 부모 = DCC 루트
DATA_DIR = BASE_DIR / "Data"
TRAINING_ROOT = DATA_DIR / "Training"  # AI-Hub 원본 Training 파티션

WORK_DIR = BASE_DIR / "Work"
TRAIN_ROOT = WORK_DIR / "Training"
WAV_OUT_DIR = TRAIN_ROOT / "wav"
JSON_OUT_DIR = TRAIN_ROOT / "json"

WAV_OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TOTAL = 29200  # 팀 공통 고정 split(train_calls.csv + val_calls.csv) 합계


print("DCC 루트로 인식된 경로:", BASE_DIR)

if not TRAINING_ROOT.exists():
    raise FileNotFoundError(
        f"경로 없음: {TRAINING_ROOT}\n"
        f"이 스크립트가 'DCC\\Code\\' 폴더 안에 있는지 확인하세요.\n"
        f"(현재 스크립트 위치: {Path(__file__).resolve()})"
    )

print("사용하는 원본 폴더:", TRAINING_ROOT)
print("(Data\\Validation 폴더는 이 스크립트에서 사용하지 않습니다)")


# ============================================================
# 파일 복사 유틸
# ============================================================

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


# ============================================================
# 실행: WAV / JSON 수집
# ============================================================

print("\nWAV 수집 중...")
wav_saved, wav_dup, wav_found = collect(TRAINING_ROOT, ".wav", WAV_OUT_DIR)

print("\nJSON 수집 중...")
json_saved, json_dup, json_found = collect(TRAINING_ROOT, ".json", JSON_OUT_DIR)


# ============================================================
# 결과 확인
# ============================================================

actual_wav = list(WAV_OUT_DIR.glob("*.wav"))
actual_json = list(JSON_OUT_DIR.glob("*.json"))

print()
print("=" * 60)
print("로컬 Training 준비 완료")
print("=" * 60)
print("발견된 WAV:", wav_found, " (신규 저장:", wav_saved, "/ 중복:", wav_dup, ")")
print("발견된 JSON:", json_found, " (신규 저장:", json_saved, "/ 중복:", json_dup, ")")

print()
print("현재 WAV 실제 개수:", len(actual_wav))
print("현재 JSON 실제 개수:", len(actual_json))

print()
print(f"기대값 (train_calls.csv + val_calls.csv 합계): {EXPECTED_TOTAL:,}")

if len(actual_wav) != EXPECTED_TOTAL or len(actual_json) != EXPECTED_TOTAL:
    print()
    print(f"경고: 개수가 {EXPECTED_TOTAL:,}과 다릅니다.")
    print("  - 더 많다면: Data\\Training 안에 서울 외 다른 지역 데이터가")
    print("    섞여 있을 수 있습니다 (원본 팀 스크립트의 '서울' 필터 참고).")
    print("  - 더 적다면: 다운로드/압축 해제가 누락된 파일이 있을 수 있습니다.")
    print("  02_common_preprocess.py 실행 시 나오는")
    print("  'missing_from_local' / 'extra_local_calls' 로그를 확인하세요.")
else:
    print(f"정상: 팀 공통 기준({EXPECTED_TOTAL:,}개)과 일치합니다.")

print()
print("WAV 위치:", WAV_OUT_DIR)
print("JSON 위치:", JSON_OUT_DIR)
