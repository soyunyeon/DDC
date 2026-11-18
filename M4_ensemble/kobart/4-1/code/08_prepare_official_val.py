import json
from pathlib import Path

ROOT = Path("/data1/workspaces/ddc/M4_ys")

SOURCE = Path(
    "/data1/workspaces/ddc/DCC_Final/Work/"
    "Qwen3_LoRA/Official_Val/"
    "official_val_prefix_all.jsonl"
)

OUTPUT = ROOT / "Work/OfficialVal/official_val_t5.jsonl"
INTERNAL = ROOT / "Work/Seq2Seq/val_prefix.jsonl"

assert SOURCE.is_file(), "공식 Validation 파일 없음"
assert INTERNAL.is_file(), "기존 T5 Validation 파일 없음"
assert not OUTPUT.exists(), "출력 파일이 이미 존재함"

count = 0
calls = set()
seen = set()
empty_text = 0

# 기존 학습 입력 형식 확인용
with INTERNAL.open(encoding="utf-8") as f:
    sample = json.loads(next(f))

with SOURCE.open(encoding="utf-8") as f:
    official_sample = json.loads(next(f))

print("===== 기존 T5 입력 예시 =====")
print(repr(sample["input_text"][:300]))

print("\n===== 공식 Validation 입력 예시 =====")
print(repr(official_sample["text"][:300]))

# 정답 라벨은 새로운 파일로 전달하지 않음
with (
    SOURCE.open(encoding="utf-8") as fi,
    OUTPUT.open("w", encoding="utf-8") as fo
):
    for line in fi:
        row = json.loads(line)

        text = row["text"]
        assert isinstance(text, str)

        step = int(row["step"])
        assert step >= 1

        key = (row["label_file"], step)
        assert key not in seen, f"중복 Prefix: {key}"
        seen.add(key)

        if not text.strip():
            empty_text += 1

        converted = {
            "call_id": row["call_id"],
            "label_file": row["label_file"],
            "step": step,
            "input_text": text,
        }

        fo.write(
            json.dumps(
                converted,
                ensure_ascii=False
            ) + "\n"
        )

        calls.add(row["call_id"])
        count += 1

assert count == 111947, f"Prefix 불일치: {count}"
assert len(calls) == 3640, f"통화 수 불일치: {len(calls)}"

# 저장된 파일의 첫 행에 정답 필드가 없는지 검증
with OUTPUT.open(encoding="utf-8") as f:
    first = json.loads(next(f))

assert set(first) == {
    "call_id", "label_file", "step", "input_text"
}

print("\n" + "=" * 55)
print("공식 Validation 변환 완료")
print("=" * 55)
print("전체 Prefix:", count)
print("전체 통화:", len(calls))
print("빈 입력 텍스트:", empty_text)
print("정답 필드 포함: 없음")
print("저장 경로:", OUTPUT)
