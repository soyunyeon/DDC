# ============================================================
# 제출 폴더 준비용 1회성 스크립트 (학습 PC에서 한 번만 실행)
#
# 학습에 썼던 jhgan/ko-sroberta-multitask의 tokenizer와 config를
# assets/ 폴더에 저장한다. 이후 inference.py는 이 폴더만 읽으므로
# 채점 환경에서 인터넷이 없어도 실행된다.
#
# 학습 때 이미 내려받은 HuggingFace 캐시를 그대로 사용하므로
# 학습에 쓴 것과 같은 파일이 저장된다.
#
# 실행: python prepare_assets.py
# ============================================================

from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

BACKBONE_NAME = "jhgan/ko-sroberta-multitask"

OUT_DIR = Path(__file__).resolve().parent / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AutoTokenizer.from_pretrained(BACKBONE_NAME).save_pretrained(OUT_DIR)
AutoConfig.from_pretrained(BACKBONE_NAME).save_pretrained(OUT_DIR)

print("저장 완료:", OUT_DIR)

for p in sorted(OUT_DIR.iterdir()):
    print(" -", p.name)
