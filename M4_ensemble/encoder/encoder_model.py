# -*- coding: utf-8 -*-
"""
encoder_model.py : Mission 4 제출용 모델 코드 (KLUE-RoBERTa 누적 입력 증상 분류)

구성
  - 입력 만들기 함수: 학습 코드(train_encoder.py)와 한 글자도 다르지 않게 같은 방식
    발화마다 " [대원] 텍스트" / " [신고자] 텍스트"로 따로 토큰화 -> 1..t번째 발화를 이어 붙임
    -> 512 토큰을 넘으면 앞 head개 + 뒷부분만 남김 -> [CLS] ... [SEP]
  - StreamingEncoder: 발화를 하나씩 받아 지금까지 받은 발화만으로 9개 증상 확률을 낸다
    (대화 전체 길이는 알 수도 없고 쓰지도 않는다)
  - pack_checkpoint: 학습 결과 폴더(best/)를 .pt 파일 하나로 묶는다
    (가중치 + 모델 설정 + 토크나이저 파일을 모두 넣어서 인터넷 없이 로드됨)
    미션별로 다른 가중치를 쓸 수 있다: 4-1 = soft Jaccard로 학습한 모델, 4-2 = BCE로 학습한 모델
    (같은 학습 결과를 두 미션에 쓰면 한 번만 저장)
"""
import base64
import io
import json
import os
import re
import tempfile

import numpy as np

SYMPTOM_CLASSES = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
N_CLS = len(SYMPTOM_CLASSES)
SPEAKER_TAG = {0: "[대원]", 1: "[신고자]"}
CKPT_FORMAT = "dcc_m4_encoder_v2"
OLD_FORMATS = ("dcc_m4_encoder_v1",)
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


# ------------------------------------------------------------
# 1. 입력 만들기 (train_encoder.py와 동일)
# ------------------------------------------------------------

def clean_text(text):
    """앞뒤 공백 제거, 연속 공백/탭을 공백 하나로."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    return text


def format_utt(speaker, text):
    return f"{SPEAKER_TAG.get(int(speaker), '[미상]')} {clean_text(text)}"


def tokenize_utt(tok, speaker, text):
    """발화 1개를 토큰 id 리스트로. 앞 공백은 발화 경계를 단어 경계로 만들기 위함."""
    return tok([" " + format_utt(speaker, text)], add_special_tokens=False)["input_ids"][0]


def special_wrap(tok):
    """토크나이저가 문장 앞뒤에 붙이는 특수 토큰을 직접 알아낸다 (예: [CLS] ... [SEP])."""
    a = tok("가나다", add_special_tokens=False)["input_ids"]
    b = tok("가나다", add_special_tokens=True)["input_ids"]
    for i in range(len(b) - len(a) + 1):
        if b[i:i + len(a)] == a:
            pre, suf = b[:i], b[i + len(a):]
            if pre or suf:
                return pre, suf
            break
    if tok.cls_token_id is not None and tok.sep_token_id is not None:
        return [tok.cls_token_id], [tok.sep_token_id]
    return [], []


def build_input(pieces, pre, suf, max_len, head):
    """발화 토큰 리스트들을 이어 붙여 모델 입력 id를 만든다. 넘치면 앞 head개 + 뒤쪽만 남김."""
    flat = [i for p in pieces for i in p]
    budget = max_len - len(pre) - len(suf)
    if len(flat) > budget:
        flat = flat[:head] + flat[len(flat) - (budget - head):]
    return pre + flat + suf


def load_tokenizer(path):
    """AutoTokenizer가 실패하면 tokenizer.json을 직접 읽는다 (train_encoder.py와 동일한 순서)."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(path)
    except Exception:                             # noqa: BLE001
        return PreTrainedTokenizerFast.from_pretrained(path)


# ------------------------------------------------------------
# 2. 체크포인트 묶기 / 풀기
# ------------------------------------------------------------

def _pack_entry(best_dir, max_len, head, info):
    """best/ 폴더 하나 -> {설정/토크나이저 파일(base64), 가중치, 입력 길이 설정}"""
    from transformers import AutoModelForSequenceClassification

    files = {}
    for fn in sorted(os.listdir(best_dir)):
        full = os.path.join(best_dir, fn)
        if os.path.isfile(full) and not fn.endswith(WEIGHT_EXT):
            with open(full, "rb") as f:
                files[fn] = base64.b64encode(f.read()).decode("ascii")
    if "config.json" not in files:
        raise SystemExit(f"[error] {best_dir} 에 config.json이 없습니다")
    model = AutoModelForSequenceClassification.from_pretrained(best_dir)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    return {"files": files, "state_dict": state, "max_len": int(max_len), "head": int(head), "info": info or {}}


def pack_checkpoint(models, out_path, halt_t_min, halt_k):
    """models: {"4-1": (best_dir, max_len, head, info), "4-2": (...)} -> .pt 파일 하나."""
    import torch

    entries, mission_map, seen = {}, {}, {}
    for mission, (best_dir, max_len, head, info) in models.items():
        key = os.path.abspath(best_dir)
        if key not in seen:                       # 같은 학습 결과는 한 번만 저장
            seen[key] = f"m{len(seen)}"
            entries[seen[key]] = _pack_entry(best_dir, max_len, head, info)
        mission_map[mission] = seen[key]
    ckpt = {
        "format": CKPT_FORMAT,
        "classes": SYMPTOM_CLASSES,
        "entries": entries,                       # 학습 결과별 가중치 + 설정 + 토크나이저
        "mission_map": mission_map,               # 미션 -> 쓸 학습 결과
        "thr": 0.5,                               # 판정 기준 (규정: 항상 0.5)
        "halt_t_min": int(halt_t_min),            # 4-2: 이 시점부터 중단 가능
        "halt_k": int(halt_k),                    # 4-2: 0.5 이상 증상 집합이 k번 연속 같으면 중단
    }
    tmp = str(out_path) + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, out_path)
    return ckpt


def _entry_for(ckpt, mission):
    if ckpt.get("format") in OLD_FORMATS:         # 예전 형식: 모델 하나를 두 미션에 공통으로
        return ckpt
    return ckpt["entries"][ckpt["mission_map"][mission]]


class StreamingEncoder:
    """발화를 하나씩 받아 그때까지의 발화만으로 증상 확률을 내는 모델."""

    def __init__(self, model, tok, device, amp_dtype, cfg, entry):
        self.model, self.tok, self.device, self.amp_dtype = model, tok, device, amp_dtype
        self.max_len, self.head = entry["max_len"], entry["head"]
        self.info = entry.get("info", {})
        self.thr = cfg["thr"]
        self.halt_t_min, self.halt_k = cfg["halt_t_min"], cfg["halt_k"]
        self.pre, self.suf = special_wrap(tok)
        self.reset()

    @classmethod
    def load(cls, ckpt_path, device="cpu", mission="4-1"):
        """체크포인트에서 해당 미션의 모델을 꺼내 로드한다 (인터넷 불필요)."""
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        if ckpt.get("format") not in (CKPT_FORMAT,) + OLD_FORMATS:
            raise ValueError(f"{ckpt_path}: 이 모델용 체크포인트가 아닙니다")
        if list(ckpt["classes"]) != SYMPTOM_CLASSES:
            raise ValueError("클래스 순서가 다릅니다")
        entry = _entry_for(ckpt, mission)
        with tempfile.TemporaryDirectory() as d:      # 설정/토크나이저 파일을 임시 폴더에 풀어 오프라인 로드
            for fn, b64 in entry["files"].items():
                with open(os.path.join(d, fn), "wb") as f:
                    f.write(base64.b64decode(b64))
            config = AutoConfig.from_pretrained(d)
            tok = load_tokenizer(d)
        model = AutoModelForSequenceClassification.from_config(config)
        model.load_state_dict(entry["state_dict"])
        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        model.to(dev).eval()
        amp = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else None
        return cls(model, tok, dev, amp, ckpt, entry)

    def reset(self):
        """새 통화 시작: 지금까지 받은 발화를 모두 비운다."""
        self.pieces = []
        self.last_probs = None

    @property
    def n_seen(self):
        return len(self.pieces)

    def add(self, speaker, text):
        """발화 1개 추가 -> 지금까지 받은 발화 1..t로 만든 입력의 9개 증상 확률 (numpy, shape (9,))."""
        import torch
        self.pieces.append(tokenize_utt(self.tok, speaker, text))
        ids = build_input(self.pieces, self.pre, self.suf, self.max_len, self.head)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                             enabled=self.amp_dtype is not None):
            logits = self.model(input_ids=x, attention_mask=torch.ones_like(x)).logits
        self.last_probs = torch.sigmoid(logits.float())[0].cpu().numpy().astype(np.float64)
        return self.last_probs

    def labels(self, probs=None):
        """확률 0.5 이상인 증상 (규정: 판정 기준 0.5 고정, 없으면 빈 리스트)."""
        p = self.last_probs if probs is None else probs
        if p is None:
            return []
        return [c for c, v in zip(SYMPTOM_CLASSES, p) if v >= self.thr]


def describe(ckpt_path):
    """체크포인트 정보 요약 (가중치 제외)."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    out = {k: ckpt.get(k) for k in ("format", "thr", "halt_t_min", "halt_k")}
    if ckpt.get("format") in OLD_FORMATS:
        ents, mm = {"m0": ckpt}, {"4-1": "m0", "4-2": "m0"}
    else:
        ents, mm = ckpt["entries"], ckpt["mission_map"]
    out["mission_map"] = mm
    out["entries"] = {k: {"max_len": e["max_len"], "head": e["head"], "info": e.get("info"),
                          "files": sorted(e["files"]),
                          "n_params": int(sum(v.numel() for v in e["state_dict"].values()))}
                      for k, e in ents.items()}
    buf = io.StringIO()
    json.dump(out, buf, ensure_ascii=False, indent=2)
    return buf.getvalue()
