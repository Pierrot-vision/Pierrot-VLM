"""공개 Qwen3-VL 체크포인트 로딩.

HuggingFace Hub 에서 모델을 자동 다운로드해 스크래치 구현 모델에 가중치를 적재한다.
모듈/파라미터 이름을 HF Qwen3-VL 키와 일치시켜 두었으므로
`load_state_dict(strict=False)` 로 바로 로드된다:
    model.visual.* / model.language_model.* / lm_head.weight(=tie)
이 로더 덕분에 공개 사전학습 가중치로 즉시 추론하고, 거기서 파인튜닝을 시작할 수 있다.

★ SmolVLM2 로더와 달리 HF AutoConfig 에 의존하지 않는다. Qwen3-VL 의 config.json 은
  모든 구조 키를 명시하므로 raw JSON 파싱만으로 정확히 복원되며, 덕분에 Qwen3-VL 을
  아직 모르는 transformers(<4.57) 환경에서도 이 구현이 그대로 동작한다.
  (토크나이저는 Qwen2 계열 BPE 라 예전 transformers 로도 로드된다.)
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional, Tuple

import torch

from .config import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig, _filter
from .modeling.qwen3vl import Qwen3VLForConditionalGeneration
from .processor import Qwen3VLProcessor

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


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
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "vocab.json", "merges.txt"],
    )


# ------------------------------------------------------------------ #
# 체크포인트의 config.json 을 읽어 Qwen3VLConfig 로 만든다.
# 두 형태를 모두 받는다:
#   - 공식 체크포인트: text_config 의 M-RoPE 설정이 rope_scaling 안에 들어 있다
#     ({"mrope_section": [...], "mrope_interleaved": true, "rope_type": "default"}).
#   - 우리 학습 산출물: engine.save_pretrained 가 asdict 로 전 필드를 평탄하게 적는다.
# ------------------------------------------------------------------ #
def config_from_json(model_dir: str) -> Qwen3VLConfig:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)

    text_raw   = dict(raw.get("text_config", {}))
    vision_raw = dict(raw.get("vision_config", {}))

    # 공식 config 의 rope_scaling 을 평탄한 필드로 옮긴다(우리 dataclass 표현).
    rope = text_raw.pop("rope_scaling", None) or {}
    if "mrope_section" in rope:
        text_raw.setdefault("mrope_section", rope["mrope_section"])
    if "mrope_interleaved" in rope:
        text_raw.setdefault("mrope_interleaved", rope["mrope_interleaved"])

    top = _filter(Qwen3VLConfig, raw)
    top["text_config"]   = Qwen3VLTextConfig(**_filter(Qwen3VLTextConfig, text_raw))
    top["vision_config"] = Qwen3VLVisionConfig(**_filter(Qwen3VLVisionConfig, vision_raw))
    return Qwen3VLConfig(**top)


# ------------------------------------------------------------------ #
# 디렉토리의 가중치를 하나의 state_dict 로 읽는다.
#   - *.safetensors 여러 샤드를 합침(공개 체크포인트)
#   - 없으면 model.pt 로드(파인튜닝 결과 디렉토리, engine.save_pretrained)
# ------------------------------------------------------------------ #
def _load_state_dict(model_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        pt = os.path.join(model_dir, "model.pt")
        if os.path.exists(pt):
            return torch.load(pt, map_location="cpu")
        raise FileNotFoundError(f"{model_dir} 에 *.safetensors 도 model.pt 도 없습니다.")
    tensors = {}
    from safetensors import safe_open

    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


# ------------------------------------------------------------------ #
# 체크포인트의 토크나이저를 로드해 Qwen3VLProcessor 를 만든다(우측 패딩).
# 픽셀 예산(min_pixels/max_pixels)·system_prompt 우선순위:
#   ① 호출자 명시(proc_kwargs 값 not None)
#   ② 우리 산출물 sidecar(qwen3vl_preprocessor.json)
#   ③ 앱 폴백(fallback, 예: args 의 안전 기본값 — sidecar 없는 공식 base 모델용)
#   ④ 공식 preprocessor_config.json 의 size(shortest/longest_edge)
# ★ fallback 이 sidecar 보다 뒤(③)라, sidecar 있는 파인튜닝 모델은 ②가 이기고
#   sidecar 없는 공식 base 모델만 ③을 쓴다. 이 판단이 다운로드 후(model_dir 확정)
#   일어나므로 로컬 경로든 Hub repo id 든 동일하게 동작한다.
# ------------------------------------------------------------------ #
def build_processor(model_dir: str, config: Qwen3VLConfig,
                    fallback: Optional[dict] = None, **proc_kwargs) -> Qwen3VLProcessor:
    from transformers import AutoTokenizer

    sidecar = _read_sidecar(model_dir)
    # sidecar 의 구조 값(patch/merge)이 이 체크포인트 config 와 다르면 잘못된 조합이다
    # (예: 다른 모델의 sidecar 를 잘못 복사). 픽셀 예산을 상속하기 전에 조기 차단한다.
    _validate_sidecar_structure(sidecar, config)
    for key in ("min_pixels", "max_pixels", "system_prompt"):
        if proc_kwargs.get(key) is None and sidecar.get(key) is not None:
            proc_kwargs[key] = sidecar[key]
    if fallback:
        for key, val in fallback.items():
            if proc_kwargs.get(key) is None and val is not None:
                proc_kwargs[key] = val
    official = _official_pixel_budget(model_dir)
    for key, val in official.items():
        if proc_kwargs.get(key) is None:
            proc_kwargs[key] = val

    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
    return Qwen3VLProcessor(tokenizer, config, **proc_kwargs)


# ------------------------------------------------------------------ #
# sidecar 의 구조 값(patch_size/merge_size)이 체크포인트 config 와 일치하는지 검증한다.
# 값이 없으면(구버전 sidecar) 통과, 다르면 잘못된 전처리 조합이므로 예외.
# ------------------------------------------------------------------ #
def _validate_sidecar_structure(sidecar: dict, config: Qwen3VLConfig) -> None:
    if not sidecar:
        return
    checks = (("patch_size", config.vision_config.patch_size),
              ("merge_size", config.vision_config.spatial_merge_size))
    for key, expected in checks:
        val = sidecar.get(key)
        if val is not None and val != expected:
            raise ValueError(
                f"[qwen3vl:weights] sidecar {key}({val}) 가 체크포인트 config "
                f"{key}({expected}) 와 다릅니다 — 잘못된 모델/전처리 조합입니다."
            )


# ------------------------------------------------------------------ #
# 우리 학습 산출물의 qwen3vl_preprocessor.json 을 읽는다(없으면 {}).
# 존재하는데 손상됐으면 조용히 넘기지 않고 예외를 낸다(전처리 불일치 조기 감지).
# ------------------------------------------------------------------ #
def _read_sidecar(model_dir: str) -> dict:
    path = os.path.join(model_dir, "qwen3vl_preprocessor.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ #
# 공식 preprocessor_config.json 의 size 에서 픽셀 예산을 읽는다.
#   size.shortest_edge → min_pixels, size.longest_edge → max_pixels
# 파일이 없으면 {} (프로세서 기본값 사용), 손상됐으면 JSONDecodeError 전파.
# ------------------------------------------------------------------ #
def _official_pixel_budget(model_dir: str) -> dict:
    path = os.path.join(model_dir, "preprocessor_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        pp = json.load(f)                                     # 손상 시 JSONDecodeError 전파
    size = pp.get("size") or {}
    out  = {}
    if size.get("shortest_edge"):
        out["min_pixels"] = int(size["shortest_edge"])
    if size.get("longest_edge"):
        out["max_pixels"] = int(size["longest_edge"])
    return out


# ------------------------------------------------------------------ #
# 공개(HF Hub)/로컬 체크포인트에서 (model, processor) 를 로드한다.
# 흐름: 경로 해석(필요시 다운로드) → config.json → 프로세서 →
#       <|image_pad|> id 를 config 에 반영 → 스크래치 모델 생성 →
#       state_dict 로드(strict=False) → (필요시) weight tying → dtype/device 이동.
#
# proc_kwargs         : 호출자가 명시한 전처리 값(최우선). None 이면 sidecar/공식값 상속.
# fallback_proc_kwargs: sidecar 가 못 채운 키에만 적용할 앱 기본값(base 모델 안전값).
# ------------------------------------------------------------------ #
def load_pretrained(
    model_id_or_path: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    fallback_proc_kwargs: Optional[dict] = None,
    **proc_kwargs,
) -> Tuple[Qwen3VLForConditionalGeneration, Qwen3VLProcessor]:
    model_dir = resolve_model_dir(model_id_or_path, revision=revision, cache_dir=cache_dir)
    config    = config_from_json(model_dir)
    processor = build_processor(model_dir, config, fallback=fallback_proc_kwargs, **proc_kwargs)

    # 프로세서가 확정한 <|image_pad|> 토큰 id 를 config 에 반영(병합 정확성 보장).
    config.image_token_id = processor.image_token_id

    model               = Qwen3VLForConditionalGeneration(config)
    state               = _load_state_dict(model_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Qwen3-VL-2B 는 tie 모델 — 체크포인트에 lm_head.weight 가 아예 없다(임베딩 공유).
    if "lm_head.weight" in missing and config.tie_word_embeddings:
        model.tie_weights()
    _verify_load(model, state, missing, unexpected)

    if dtype is not None:
        model = model.to(dtype)
    model = model.to(device)
    return model, processor


# ------------------------------------------------------------------ #
# 가중치 없이 config 만으로 모델 생성(랜덤 초기화, 스크래치 학습용).
# ------------------------------------------------------------------ #
def build_model_from_config(config: Qwen3VLConfig) -> Qwen3VLForConditionalGeneration:
    model = Qwen3VLForConditionalGeneration(config)
    if config.tie_word_embeddings:
        model.tie_weights()
    return model


# ------------------------------------------------------------------ #
# 로드 검증(엄격): 사전학습 체크포인트를 로드할 때는 조용한 랜덤 초기화를 막는다.
#   - tie_word_embeddings=True 일 때의 lm_head.weight 누락만 허용(tie 로 채워짐).
#   - 그 외 missing(=일부 레이어 랜덤 초기화 위험)/unexpected(=포맷 불일치)는 예외.
#   - 로드된 파라미터 텐서 비율을 출력해 실제 적재량을 확인하게 한다.
# (shape mismatch 는 load_state_dict 가 이미 예외를 낸다.)
# ------------------------------------------------------------------ #
def _verify_load(model, state, missing, unexpected) -> None:
    allowed_missing = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    total  = len(model.state_dict())
    loaded = total - len(missing)
    print(f"[qwen3vl:weights] 체크포인트에서 로드된 텐서 키: {loaded}/{total} "
          f"(체크포인트 제공 {len(state)}개)")

    hard_missing = [m for m in missing if m not in allowed_missing]
    if hard_missing:
        raise RuntimeError(
            f"[qwen3vl:weights] 누락 키 {len(hard_missing)}개(해당 레이어가 랜덤 초기화됨): "
            f"{hard_missing[:8]} ... config 와 체크포인트가 일치하는지 확인하세요."
        )
    if unexpected:
        raise RuntimeError(
            f"[qwen3vl:weights] 예상 밖 키 {len(unexpected)}개: {unexpected[:8]} ... "
            f"체크포인트 포맷/디렉토리를 확인하세요."
        )
