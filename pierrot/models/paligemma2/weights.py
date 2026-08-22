"""공개 PaliGemma2 체크포인트 로딩.

HuggingFace Hub 에서 모델을 자동 다운로드해 스크래치 구현 모델에 가중치를 적재한다.
모듈/파라미터 이름을 HF 키와 일치시켜 두었으므로 `load_state_dict(strict=False)` 로
바로 로드된다. 이 로더 덕분에:
  - 공개 사전학습 가중치로 즉시 추론 가능
  - 사전학습 가중치에서 파인튜닝 시작 가능
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional, Tuple

import torch

from .config import Gemma2Config, PaliGemma2Config, SiglipConfig, _filter
from .modeling.paligemma2 import PaliGemma2ForConditionalGeneration
from .processor import PaliGemma2Processor

DEFAULT_MODEL_ID = "google/paligemma2-3b-pt-224"


# ------------------------------------------------------------------ #
# 로컬 경로면 그대로, Hub id 면 다운로드 후 로컬 경로를 반환.
# ------------------------------------------------------------------ #
def resolve_model_dir(model_id_or_path: str, revision: Optional[str] = None,
                      cache_dir: Optional[str] = None) -> str:
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id_or_path,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"],
    )


# ------------------------------------------------------------------ #
# 체크포인트의 config.json 을 읽어 PaliGemma2Config 로 만든다.
# text_config/vision_config 의 HF 여분 키는 _filter 로 걸러 우리 필드만 남긴다.
# ------------------------------------------------------------------ #
def config_from_json(model_dir: str) -> PaliGemma2Config:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    text                 = raw.get("text_config", {})
    vision               = raw.get("vision_config", {})
    top                  = _filter(PaliGemma2Config, raw)
    top["text_config"]   = Gemma2Config(**_filter(Gemma2Config, text))
    top["vision_config"] = SiglipConfig(**_filter(SiglipConfig, vision))
    return PaliGemma2Config(**top)


# ------------------------------------------------------------------ #
# 디렉토리의 가중치를 하나의 state_dict 로 읽는다.
#   - *.safetensors 여러 샤드를 합침(공개 체크포인트)
#   - 없으면 model.pt 로드(파인튜닝 결과 디렉토리)
# 최신 transformers 가 붙이는 'model.' 접두어는 제거해 우리 키에 맞춘다.
# ------------------------------------------------------------------ #
def _load_state_dict(model_dir: str) -> dict:
    tensors = {}
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        # 파인튜닝 결과 디렉토리(engine.save_pretrained)는 model.pt 로 저장된다.
        pt = os.path.join(model_dir, "model.pt")
        if os.path.exists(pt):
            return torch.load(pt, map_location="cpu")
        raise FileNotFoundError(f"{model_dir} 에 *.safetensors 도 model.pt 도 없습니다.")
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    # 최신 transformers 는 'model.' 접두어를 붙여 저장하기도 함 -> 제거해 키를 맞춘다.
    if any(k.startswith("model.") for k in tensors):
        remapped = {}
        for k, v in tensors.items():
            nk           = k[len("model."):] if k.startswith("model.") and not k.startswith("model.embed") else k
            remapped[nk] = v
        tensors = remapped
    return tensors


# ------------------------------------------------------------------ #
# 체크포인트의 토크나이저를 로드해 PaliGemma2Processor 를 만든다(우측 패딩).
# 이미지 크기/토큰 수는 config 에서 가져온다.
# ------------------------------------------------------------------ #
def build_processor(model_dir: str, config: PaliGemma2Config) -> PaliGemma2Processor:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
    return PaliGemma2Processor(
        tokenizer,
        image_size=config.vision_config.image_size,
        num_image_tokens=config.vision_config.num_patches,
    )


# ------------------------------------------------------------------ #
# 공개(HF Hub)/로컬 체크포인트에서 (model, processor) 를 로드한다.
# 흐름: 경로 해석(필요시 다운로드) → config.json → 프로세서 →
#       <image> id 를 config 에 반영 → 스크래치 모델 생성 →
#       state_dict 로드(strict=False) → weight tying → dtype/device 이동.
# 사전학습 추론과 파인튜닝 시작점 로딩에 공용으로 쓰인다.
# ------------------------------------------------------------------ #
def load_pretrained(
    model_id_or_path: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[PaliGemma2ForConditionalGeneration, PaliGemma2Processor]:
    model_dir = resolve_model_dir(model_id_or_path, revision=revision, cache_dir=cache_dir)
    config    = config_from_json(model_dir)
    processor = build_processor(model_dir, config)

    # 프로세서가 확정한 <image> 토큰 id 를 config 에 반영(병합 정확성 보장).
    config.image_token_index = processor.image_token_id

    model               = PaliGemma2ForConditionalGeneration(config)
    state               = _load_state_dict(model_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.tie_weights()

    # tie 이후 lm_head.weight 는 embed 와 공유되므로 missing 목록의 그 항목은 무해.
    _report_load(missing, unexpected)

    if dtype is not None:
        model = model.to(dtype)
    model = model.to(device)
    return model, processor


# ------------------------------------------------------------------ #
# Accelerate save_state 체크포인트(checkpoint-<step>/)로 추론 모델을 만든다.
# 그 디렉토리엔 model.safetensors 만 있고 config.json·토크나이저가 없으므로,
# 구조/토크나이저는 base(원 사전학습 모델)에서 가져오고 가중치만 체크포인트로 덮어쓴다.
# tie 로 저장에서 빠진 lm_head 는 tie_weights() 로 복원된다.
# 학습이 끝나기 전에 중간 체크포인트로 바로 추론할 때 쓴다.
# ------------------------------------------------------------------ #
def load_from_checkpoint(
    ckpt_dir: str,
    base: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[PaliGemma2ForConditionalGeneration, PaliGemma2Processor]:
    base_dir  = resolve_model_dir(base, revision=revision, cache_dir=cache_dir)
    config    = config_from_json(base_dir)
    processor = build_processor(base_dir, config)
    config.image_token_index = processor.image_token_id

    model               = PaliGemma2ForConditionalGeneration(config)
    state               = _load_state_dict(ckpt_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.tie_weights()
    _report_load(missing, unexpected)

    if dtype is not None:
        model = model.to(dtype)
    return model.to(device), processor


# ------------------------------------------------------------------ #
# 가중치 없이 config 만으로 모델 생성(랜덤 초기화, 스크래치 학습용).
# ------------------------------------------------------------------ #
def build_model_from_config(config: PaliGemma2Config) -> PaliGemma2ForConditionalGeneration:
    model = PaliGemma2ForConditionalGeneration(config)
    model.tie_weights()
    return model


# ------------------------------------------------------------------ #
# load_state_dict 의 missing/unexpected 키를 요약 출력한다.
# lm_head.weight 는 tie 로 채워지므로 missing 경고에서 제외한다.
# ------------------------------------------------------------------ #
def _report_load(missing, unexpected) -> None:
    # lm_head.weight 는 tie 로 채워지므로 경고에서 제외.
    missing = [m for m in missing if m != "language_model.lm_head.weight"]
    if missing:
        print(f"[weights] load_state_dict missing keys ({len(missing)}): {missing[:8]}...")
    if unexpected:
        print(f"[weights] load_state_dict unexpected keys ({len(unexpected)}): {unexpected[:8]}...")
