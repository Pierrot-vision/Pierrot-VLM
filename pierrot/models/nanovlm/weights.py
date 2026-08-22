"""nanoVLM 가중치 로딩 (추론 전용).

체크포인트 하나를 읽어 (model, processor) 를 만든다:
  - 공개 nanoVLM 체크포인트 (config.json + model.safetensors)
  - 우리 학습 산출물 final  (config.json + model.pt)

백본(SigLIP2+SmolLM2)에서 시작하는 학습 초기화 경로는 학습 저장소 Pierrot-VLM-Lab 쪽에 있다.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional, Tuple

import torch

from .config import VLMConfig
from .modeling.vision_language_model import VisionLanguageModel
from .processor import NanoVLMProcessor

DEFAULT_MODEL_ID = "lusxvr/nanoVLM-450M"


# ------------------------------------------------------------------ #
# 로컬 경로면 그대로, Hub id 면 다운로드 후 로컬 경로를 반환.
# ------------------------------------------------------------------ #
def resolve_model_dir(model_id_or_path: str, revision: Optional[str] = None,
                      cache_dir: Optional[str] = None) -> str:
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id_or_path, revision=revision, cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.model", "tokenizer*", "*.txt"],
    )


# ------------------------------------------------------------------ #
# 디렉토리에서 state_dict 를 읽는다: model.safetensors 우선, 없으면 model.pt.
# ------------------------------------------------------------------ #
def _load_state_dict(model_dir: str) -> dict:
    st = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if st:
        from safetensors import safe_open
        tensors = {}
        for path in sorted(st):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
        return tensors
    pt = os.path.join(model_dir, "model.pt")
    if os.path.exists(pt):
        return torch.load(pt, map_location="cpu")
    raise FileNotFoundError(f"{model_dir} 에 model.safetensors 도 model.pt 도 없습니다.")


# ------------------------------------------------------------------ #
# cfg 로 프로세서를 만들고 모델에 <|image|>/eos 토큰 id 를 주입한다.
# (병합·생성에 필요 — 모델은 토크나이저를 직접 들지 않는다)
# ------------------------------------------------------------------ #
def _bind_processor(model: VisionLanguageModel, cfg: VLMConfig) -> NanoVLMProcessor:
    processor = NanoVLMProcessor(cfg)
    model.set_image_token_id(processor.image_token_id)
    model._eos_token_id = processor.eos_token_id
    return model, processor  # type: ignore[return-value]


# ------------------------------------------------------------------ #
# 공개/로컬 VLM 체크포인트에서 (model, processor) 를 로드한다(추론/파인튜닝 공용).
# 흐름: 경로 해석(필요시 다운로드) → config.json → 모델 생성(백본 미로드) →
#       state_dict 로드(strict=False) → tie → 프로세서 바인딩 → dtype/device.
# ------------------------------------------------------------------ #
def load_pretrained(
    model_id_or_path: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[VisionLanguageModel, NanoVLMProcessor]:
    model_dir = resolve_model_dir(model_id_or_path, revision=revision, cache_dir=cache_dir)
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = VLMConfig(**_filter(VLMConfig, json.load(f)))

    model               = VisionLanguageModel(cfg, load_backbone=False)
    state               = _load_state_dict(model_dir)
    # 공식 nanoVLM safetensors 는 tie 된 임베딩을 dedup 해 decoder.head.weight 만 저장하고
    # decoder.token_embedding.weight 를 뺀다(반대인 경우도 있음). 로드 전에 서로 미러링해
    # 어느 쪽도 랜덤 초기화되지 않게 한다(우리 model.pt 는 둘 다 있어 무해).
    if cfg.lm_tie_weights:
        if "decoder.token_embedding.weight" not in state and "decoder.head.weight" in state:
            state["decoder.token_embedding.weight"] = state["decoder.head.weight"]
        elif "decoder.head.weight" not in state and "decoder.token_embedding.weight" in state:
            state["decoder.head.weight"] = state["decoder.token_embedding.weight"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.tie_weights()
    _verify_load(missing, unexpected)

    model, processor = _bind_processor(model, cfg)
    if dtype is not None:
        model = model.to(dtype)
    model = model.to(device)
    return model, processor


# ------------------------------------------------------------------ #
# dataclass 필드에 존재하는 키만 남긴다(config.json 여분 키 무시).
# ------------------------------------------------------------------ #
def _filter(cls, d: dict) -> dict:
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in d.items() if k in valid}


# ------------------------------------------------------------------ #
# 로드 검증: tie 로 채워지는 decoder.head.weight 누락만 무해로 허용하고,
# 그 외 missing(=랜덤 초기화 위험)/unexpected(=포맷 불일치·이물 safetensors)는
# 예외로 처리한다. (shape mismatch 는 load_state_dict 가 이미 예외를 낸다)
# ------------------------------------------------------------------ #
def _verify_load(missing, unexpected) -> None:
    missing = [m for m in missing if m != "decoder.head.weight"]
    if unexpected:
        raise RuntimeError(
            f"[nanovlm:weights] 예상 밖 키 {len(unexpected)}개: {unexpected[:8]} ... "
            f"체크포인트 포맷/디렉토리(이물 *.safetensors 여부)를 확인하세요."
        )
    if missing:
        raise RuntimeError(
            f"[nanovlm:weights] 누락 키 {len(missing)}개(해당 레이어가 랜덤 초기화됨): {missing[:8]} ... "
            f"config 와 체크포인트가 일치하는지 확인하세요."
        )
