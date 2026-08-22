"""공개 Qwen3.5 체크포인트 로딩 (스크래치 모델용).

HuggingFace Hub 에서 모델을 자동 다운로드해 스크래치 구현 모델에 가중치를 적재한다.
모듈/파라미터 이름을 HF Qwen3.5 키와 일치시켜 두었으므로
`load_state_dict(strict=False)` 로 바로 로드된다:
    model.visual.* / model.language_model.* / lm_head.weight(=tie)

★ qwen3vl 로더와 같은 원칙 — HF AutoConfig 에 의존하지 않는다. Qwen3.5 의
  config.json 도 모든 구조 키를 명시하므로 raw JSON 파싱만으로 정확히 복원되며,
  덕분에 Qwen3.5 를 아직 모르는 transformers(<5.1) 환경에서도 그대로 동작한다.
  (토크나이저는 Qwen2 계열 BPE 라 예전 transformers 로도 로드된다.)

Qwen3.5 만의 처리 두 가지:
  - 공식 config 는 RoPE 설정이 text_config.rope_parameters 안에 있다
    (rope_theta / partial_rotary_factor / mrope_section / mrope_interleaved)
    → 평탄한 dataclass 필드로 승격한다.
  - 공식 체크포인트에 **mtp.*** (multi-token prediction 보조 헤드) 가중치가 있다.
    본 모델과 무관하므로 로드 전에 걸러낸다(공식 구현도 무시하는 키).
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import torch

# 경로 해석·safetensors 적재·sidecar/픽셀 예산 읽기는 qwen3vl 로더와 완전히 같은
# 규약이라 그대로 재사용한다(sidecar 파일명도 프로세서 상속으로 동일).
from ..qwen3vl.weights import (
    _load_state_dict,
    _official_pixel_budget,
    _read_sidecar,
    _validate_sidecar_structure,
    resolve_model_dir,
)
from .config import Qwen35Config, Qwen35TextConfig, Qwen35VisionConfig, _filter
from .modeling.qwen35 import Qwen35ForConditionalGeneration
from .processor import Qwen35Processor

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"


# ------------------------------------------------------------------ #
# 체크포인트의 config.json 을 읽어 Qwen35Config 로 만든다.
# 두 형태를 모두 받는다:
#   - 공식 체크포인트: RoPE 설정이 text_config.rope_parameters 안에 있다.
#   - 우리 학습 산출물: engine.save_pretrained 가 asdict 로 전 필드를 평탄하게 적는다.
# ------------------------------------------------------------------ #
def config_from_json(model_dir: str) -> Qwen35Config:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)

    text_raw   = dict(raw.get("text_config", {}))
    vision_raw = dict(raw.get("vision_config", {}))

    # 공식 config 의 rope_parameters(또는 구형 rope_scaling)를 평탄한 필드로 옮긴다.
    rope = text_raw.pop("rope_parameters", None) or text_raw.pop("rope_scaling", None) or {}
    for key in ("rope_theta", "partial_rotary_factor", "mrope_section", "mrope_interleaved"):
        if key in rope:
            text_raw.setdefault(key, rope[key])

    top = _filter(Qwen35Config, raw)
    top["text_config"]   = Qwen35TextConfig(**_filter(Qwen35TextConfig, text_raw))
    top["vision_config"] = Qwen35VisionConfig(**_filter(Qwen35VisionConfig, vision_raw))
    return Qwen35Config(**top)


# ------------------------------------------------------------------ #
# 체크포인트의 토크나이저를 로드해 Qwen35Processor 를 만든다(우측 패딩).
# 픽셀 예산·system_prompt 우선순위는 qwen3vl 과 동일:
#   ① 호출자 명시 → ② 산출물 sidecar → ③ 앱 폴백 → ④ 공식 preprocessor_config.json
# ------------------------------------------------------------------ #
def build_processor(model_dir: str, config: Qwen35Config,
                    fallback: Optional[dict] = None, **proc_kwargs) -> Qwen35Processor:
    from transformers import AutoTokenizer

    sidecar = _read_sidecar(model_dir)
    _validate_sidecar_structure(sidecar, config)
    # enable_thinking(템플릿 모드)도 sidecar 로 상속 — 학습/추론 프롬프트 분포 일치.
    for key in ("min_pixels", "max_pixels", "system_prompt", "enable_thinking"):
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
    return Qwen35Processor(tokenizer, config, **proc_kwargs)


# ------------------------------------------------------------------ #
# 공개(HF Hub)/로컬 체크포인트에서 (model, processor) 를 로드한다.
# 흐름: 경로 해석(필요시 다운로드) → config.json → 프로세서 →
#       <|image_pad|> id 반영 → 스크래치 모델 생성 → mtp.* 제거 후
#       state_dict 로드(strict=False) → (필요시) weight tying → dtype/device 이동.
# ------------------------------------------------------------------ #
def load_pretrained(
    model_id_or_path: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    fallback_proc_kwargs: Optional[dict] = None,
    **proc_kwargs,
) -> Tuple[Qwen35ForConditionalGeneration, Qwen35Processor]:
    model_dir = resolve_model_dir(model_id_or_path, revision=revision, cache_dir=cache_dir)
    config    = config_from_json(model_dir)
    processor = build_processor(model_dir, config, fallback=fallback_proc_kwargs, **proc_kwargs)

    # 프로세서가 확정한 <|image_pad|> 토큰 id 를 config 에 반영(병합 정확성 보장).
    config.image_token_id = processor.image_token_id

    model = Qwen35ForConditionalGeneration(config)
    state = _load_state_dict(model_dir)
    # 공식 체크포인트의 MTP(multi-token prediction) 보조 헤드는 본 모델과 무관하다.
    dropped = [k for k in state if k.startswith("mtp.")]
    for key in dropped:
        state.pop(key)
    if dropped:
        print(f"[qwen35:weights] mtp.* 키 {len(dropped)}개 무시(멀티토큰 예측 보조 헤드).")

    missing, unexpected = model.load_state_dict(state, strict=False)
    # Qwen3.5-4B 는 tie 모델 — 체크포인트에 lm_head.weight 가 아예 없다(임베딩 공유).
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
def build_model_from_config(config: Qwen35Config) -> Qwen35ForConditionalGeneration:
    model = Qwen35ForConditionalGeneration(config)
    if config.tie_word_embeddings:
        model.tie_weights()
    return model


# ------------------------------------------------------------------ #
# 로드 검증(엄격): 사전학습 체크포인트를 로드할 때 조용한 랜덤 초기화를 막는다.
#   - tie_word_embeddings=True 일 때의 lm_head.weight 누락만 허용(tie 로 채워짐).
#   - 그 외 missing/unexpected 는 예외(mtp.* 는 이미 로드 전에 제거됨).
# ------------------------------------------------------------------ #
def _verify_load(model, state, missing, unexpected) -> None:
    allowed_missing = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    total  = len(model.state_dict())
    loaded = total - len(missing)
    print(f"[qwen35:weights] 체크포인트에서 로드된 텐서 키: {loaded}/{total} "
          f"(체크포인트 제공 {len(state)}개)")

    hard_missing = [m for m in missing if m not in allowed_missing]
    if hard_missing:
        raise RuntimeError(
            f"[qwen35:weights] 누락 키 {len(hard_missing)}개(해당 레이어가 랜덤 초기화됨): "
            f"{hard_missing[:8]} ... config 와 체크포인트가 일치하는지 확인하세요."
        )
    if unexpected:
        raise RuntimeError(
            f"[qwen35:weights] 예상 밖 키 {len(unexpected)}개: {unexpected[:8]} ... "
            f"체크포인트 포맷/디렉토리를 확인하세요."
        )
