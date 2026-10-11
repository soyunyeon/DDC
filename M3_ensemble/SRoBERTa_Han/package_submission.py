# ============================================================
# SRoBERTa 제출 폴더 완성 + 압축 (학습 PC에서 한 번만 실행)
#
# 이 파일을 SRoBERTa_Han 폴더 안에 넣고 실행:
#   python package_submission.py
#
# 하는 일
# 1. checkpoints/best_model.pt 없으면 Data/Work/M3_FT/best_model.pt에서 복사
# 2. assets/ (tokenizer, config) 없으면 HF 캐시에서 생성
# 3. 모델이 로컬 파일만으로 로드되는지 확인 (인터넷 없이)
# 4. 폴더 전체를 SRoBERTa_Han.zip으로 압축 (폴더 바로 바깥에 생성)
# ============================================================

import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

HERE = Path(__file__).resolve().parent

CKPT = HERE / "checkpoints" / "best_model.pt"
ASSETS = HERE / "assets"

BACKBONE_NAME = "jhgan/ko-sroberta-multitask"

# 체크포인트 원본 후보 (폴더 위치에 따라 자동 탐색)
CKPT_CANDIDATES = [
    HERE.parent / "best_model.pt",                                   # M3_FT/SRoBERTa_Han 인 경우
    Path.home() / "Desktop" / "DCC" / "Data" / "Work" / "M3_FT" / "best_model.pt",
]


# 1. checkpoint
if not CKPT.exists():
    src = next((p for p in CKPT_CANDIDATES if p.exists()), None)
    if src is None:
        raise FileNotFoundError("best_model.pt를 찾지 못했습니다. checkpoints/ 폴더에 직접 넣어주세요.")
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, CKPT)
    print("checkpoint 복사:", src)
else:
    print("checkpoint 확인:", CKPT)


# 2. assets
if not (ASSETS / "config.json").exists() or not (ASSETS / "tokenizer_config.json").exists():
    ASSETS.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(BACKBONE_NAME).save_pretrained(ASSETS)
    AutoConfig.from_pretrained(BACKBONE_NAME).save_pretrained(ASSETS)
    print("assets 생성:", ASSETS)
else:
    print("assets 확인:", ASSETS)


# 3. 로컬 파일만으로 로드 테스트
import sys
sys.path.insert(0, str(HERE))
from model import HierarchicalClassifier

AutoTokenizer.from_pretrained(str(ASSETS), local_files_only=True)
model = HierarchicalClassifier(ASSETS, num_labels=9)
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
print("로드 테스트 통과 (파라미터 %s개)" % format(sum(p.numel() for p in model.parameters()), ","))


# 4. 압축
zip_base = HERE.parent / HERE.name
archive = shutil.make_archive(str(zip_base), "zip", root_dir=HERE.parent, base_dir=HERE.name)
size_mb = Path(archive).stat().st_size / 1024 / 1024
print(f"압축 완료: {archive} ({size_mb:.0f}MB)")
print("이 zip을 구글 드라이브 등에 올려 공유하면 됩니다.")
