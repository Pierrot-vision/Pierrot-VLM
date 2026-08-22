"""공개 SmolVLM2 체크포인트 로딩.

HuggingFace Hub 에서 모델을 자동 다운로드해 스크래치 구현 모델에 가중치를 적재한다.
모듈/파라미터 이름을 HF SmolVLM(Idefics3) 키와 일치시켜 두었으므로
`load_state_dict(strict=False)` 로 바로 로드된다:
    model.vision_model.* / model.connector.* / model.text_model.* / lm_head.weight
이 로더 덕분에 공개 사전학습 가중치로 즉시 추론하고, 거기서 파인튜닝을 시작할 수 있다.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional, Tuple

import torch

from .config import SmolLM2TextConfig, SmolVLM2Config, SmolVLMVisionConfig, _filter
from .modeling.smolvlm2 import SmolVLM2ForConditionalGeneration
from .processor import SmolVLM2Processor

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


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
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt", "merges.txt", "vocab.json"],
    )


# ------------------------------------------------------------------ #
# 체크포인트의 config.json 을 읽어 SmolVLM2Config 로 만든다.
#
# ★ 공식 SmolVLM 체크포인트는 config.json 에서 일부 키(예: 256M/500M 의 vision
#   intermediate_size·num_hidden_layers, 2.2B 의 vision hidden_size·heads)를 생략하고
#   HF config 클래스 기본값에 의존한다(크기마다 생략 키가 다르다!). 따라서 우리 dataclass
#   기본값으로 채우면 크기별로 어긋난다 → HF AutoConfig 로 기본값까지 정확히 해석한 뒤 매핑한다.
#   현재 256M/500M/2.2B 를 정확히 로드한다(향후 아키텍처가 바뀐 체크포인트는 _config_from_hf
#   의 매핑 필드를 확장해야 할 수 있다).
#
# 분기 규칙(안전):
#   - config.json 의 model_type 이 SmolVLM 계열이면 반드시 HF 로 해석한다. AutoConfig 가
#     실패하면(구버전 transformers/손상) 조용히 raw 폴백하지 않고 명확한 예외를 낸다
#     (raw 폴백은 생략 키를 우리 기본값으로 잘못 채워 shape mismatch 를 재발시키므로).
#   - model_type 이 없으면(=우리 학습 산출물, asdict 로 전 필드 명시) raw JSON + _filter.
#     이 경우 생략 키가 없어 기본값 문제가 발생하지 않는다.
# ------------------------------------------------------------------ #
def config_from_json(model_dir: str) -> SmolVLM2Config:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)

    if raw.get("model_type") in ("smolvlm", "smolvlm2", "idefics3"):
        # 공식 체크포인트: HF 로 기본값까지 해석해야 정확하다. 실패 시 명확히 중단.
        try:
            from transformers import AutoConfig
            hf = AutoConfig.from_pretrained(model_dir)
        except Exception as e:
            raise RuntimeError(
                f"[smolvlm2:weights] '{raw.get('model_type')}' 체크포인트인데 HF AutoConfig 해석 실패: {e}. "
                f"transformers(SmolVLM 지원, ≥4.50)를 설치/업데이트하세요. "
                f"(생략된 config 키를 임의 기본값으로 채우면 가중치 shape 가 어긋납니다.)"
            ) from e
        return _config_from_hf(hf)

    # 우리 학습 산출물 등 model_type 없는 config: raw + _filter (전 필드 명시라 안전).
    text                 = raw.get("text_config", {})
    vision               = raw.get("vision_config", {})
    top                  = _filter(SmolVLM2Config, raw)
    top["text_config"]   = SmolLM2TextConfig(**_filter(SmolLM2TextConfig, text))
    top["vision_config"] = SmolVLMVisionConfig(**_filter(SmolVLMVisionConfig, vision))
    return SmolVLM2Config(**top)


# ------------------------------------------------------------------ #
# HF 로 완전 해석된 config 를 우리 dataclass 로 매핑한다(기본값까지 확정된 값 사용).
# ------------------------------------------------------------------ #
def _config_from_hf(hf) -> SmolVLM2Config:
    v, t = hf.vision_config, hf.text_config
    vision = SmolVLMVisionConfig(
        hidden_size=v.hidden_size, intermediate_size=v.intermediate_size,
        num_hidden_layers=v.num_hidden_layers, num_attention_heads=v.num_attention_heads,
        num_channels=getattr(v, "num_channels", 3), image_size=v.image_size, patch_size=v.patch_size,
        layer_norm_eps=getattr(v, "layer_norm_eps", 1e-6),
        attention_dropout=getattr(v, "attention_dropout", 0.0),
    )
    text = SmolLM2TextConfig(
        vocab_size=t.vocab_size, hidden_size=t.hidden_size, intermediate_size=t.intermediate_size,
        num_hidden_layers=t.num_hidden_layers, num_attention_heads=t.num_attention_heads,
        num_key_value_heads=getattr(t, "num_key_value_heads", None) or t.num_attention_heads,
        head_dim=getattr(t, "head_dim", None),
        max_position_embeddings=t.max_position_embeddings, rms_norm_eps=t.rms_norm_eps,
        rope_theta=t.rope_theta, attention_dropout=getattr(t, "attention_dropout", 0.0),
        pad_token_id=getattr(t, "pad_token_id", None),
    )
    return SmolVLM2Config(
        text_config=text, vision_config=vision,
        scale_factor=hf.scale_factor, image_token_id=hf.image_token_id,
        pad_token_id=getattr(hf, "pad_token_id", None),
        tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
    )


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
# 체크포인트의 토크나이저를 로드해 SmolVLM2Processor 를 만든다(우측 패딩).
# 타일 설정(do_image_splitting/size_longest_edge/max_splits_per_side) 우선순위:
#   ① 호출자 명시(proc_kwargs 값 not None)  ② 우리 산출물 sidecar(smolvlm2_preprocessor.json)
#   ③ (size 한정) 공식 preprocessor_config.json  ④ 앱 폴백(fallback, 예: base 모델용 안전 기본값)
#   ⑤ 프로세서 자동 도출(image_size×4 / 4)
# ★ fallback 은 sidecar 보다 뒤(④)라, sidecar 있는 파인튜닝 모델은 ②가 이기고, sidecar 없는
#   공식 base 모델만 ④(안전 기본값)를 쓴다. 이 판단이 다운로드 후(model_dir 확정 시점)에
#   일어나므로 로컬 경로든 Hub repo id 든 동일하게 동작한다.
# ------------------------------------------------------------------ #
def build_processor(model_dir: str, config: SmolVLM2Config,
                    fallback: Optional[dict] = None, **proc_kwargs) -> SmolVLM2Processor:
    from transformers import AutoTokenizer

    # ② sidecar: 호출자가 명시하지 않은 키만 채우고 image_size 정합성 검증.
    proc_kwargs = _merge_sidecar(proc_kwargs, _read_sidecar(model_dir), config.vision_config.image_size)
    # ③ size 는 공식 preprocessor_config.json 을 폴백으로 사용.
    if proc_kwargs.get("size_longest_edge") is None:
        official = _official_longest_edge(model_dir)
        if official is not None:
            proc_kwargs["size_longest_edge"] = official
    # ④ 앱 폴백(sidecar/공식이 못 채운 키만): 공식 base 모델용 안전 기본값 등.
    if fallback:
        for key, val in fallback.items():
            if proc_kwargs.get(key) is None and val is not None:
                proc_kwargs[key] = val

    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
    return SmolVLM2Processor(tokenizer, config, **proc_kwargs)


# ------------------------------------------------------------------ #
# sidecar 값을 proc_kwargs 에 병합한다(호출자가 명시하지 않은 None 키만).
# sidecar 의 image_size 가 체크포인트 config 와 다르면 잘못된 조합이므로 예외.
# ------------------------------------------------------------------ #
def _merge_sidecar(proc_kwargs: dict, sidecar: dict, config_image_size: int) -> dict:
    if not sidecar:
        return proc_kwargs
    si = sidecar.get("image_size")
    if si is not None and si != config_image_size:
        raise ValueError(
            f"[smolvlm2:weights] sidecar image_size({si}) 가 체크포인트 config image_size"
            f"({config_image_size}) 와 다릅니다 — 잘못된 모델/전처리 조합입니다."
        )
    for key in ("do_image_splitting", "size_longest_edge", "max_splits_per_side"):
        if proc_kwargs.get(key) is None and key in sidecar:
            proc_kwargs[key] = sidecar[key]
    return proc_kwargs


# ------------------------------------------------------------------ #
# 우리 학습 산출물의 smolvlm2_preprocessor.json 을 읽는다(없으면 {}).
# 존재하는데 손상됐으면 조용히 넘기지 않고 예외를 낸다(전처리 불일치 조기 감지).
# ------------------------------------------------------------------ #
def _read_sidecar(model_dir: str) -> dict:
    path = os.path.join(model_dir, "smolvlm2_preprocessor.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ #
# 체크포인트의 preprocessor_config.json 에서 공식 size.longest_edge 를 읽는다.
# 반환 규약:
#   - 파일 없음         → None (프로세서가 image_size×4 로 도출; 정상 폴백)
#   - 파일 있고 손상됨   → JSONDecodeError 전파 (조용히 무시하지 않음)
#   - 파일 있고 키 없음  → None (image_size×4 폴백; 키 부재는 다른 포맷일 수 있어 치명 아님)
# ------------------------------------------------------------------ #
def _official_longest_edge(model_dir: str) -> Optional[int]:
    path = os.path.join(model_dir, "preprocessor_config.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        pp = json.load(f)                                     # 손상 시 JSONDecodeError 전파
    edge = (pp.get("size") or {}).get("longest_edge")
    return int(edge) if edge else None                        # 키 없음 → None(폴백)


# ------------------------------------------------------------------ #
# 공개(HF Hub)/로컬 체크포인트에서 (model, processor) 를 로드한다.
# 흐름: 경로 해석(필요시 다운로드) → config.json → 프로세서 →
#       <image> id 를 config 에 반영 → 스크래치 모델 생성 →
#       state_dict 로드(strict=False) → (필요시) weight tying → dtype/device 이동.
#
# proc_kwargs        : 호출자가 명시한 전처리 값(최우선). None 이면 sidecar/공식값 상속.
# fallback_proc_kwargs: sidecar/공식이 못 채운 키에만 적용할 앱 기본값(예: base 모델 안전값).
#   ★ 다운로드 후(model_dir 확정) build_processor 안에서 sidecar 를 먼저 보므로, Hub repo id
#     로 올린 파인튜닝 모델도 sidecar 가 fallback 을 이긴다(로컬/원격 동일 동작).
# ------------------------------------------------------------------ #
def load_pretrained(
    model_id_or_path: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    fallback_proc_kwargs: Optional[dict] = None,
    **proc_kwargs,
) -> Tuple[SmolVLM2ForConditionalGeneration, SmolVLM2Processor]:
    model_dir = resolve_model_dir(model_id_or_path, revision=revision, cache_dir=cache_dir)
    config    = config_from_json(model_dir)
    processor = build_processor(model_dir, config, fallback=fallback_proc_kwargs, **proc_kwargs)

    # 프로세서가 확정한 <image> 토큰 id 를 config 에 반영(병합 정확성 보장).
    config.image_token_id = processor.image_token_id

    model               = SmolVLM2ForConditionalGeneration(config)
    state               = _load_state_dict(model_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # SmolLM2 는 임베딩 tie 가능 — 체크포인트에 lm_head.weight 가 없으면(tie 모델) 공유로 채운다.
    # (공식 2.2B 는 tie_word_embeddings=False 라 lm_head.weight 가 실제로 존재해 이 경로를 안 탐.)
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
def build_model_from_config(config: SmolVLM2Config) -> SmolVLM2ForConditionalGeneration:
    model = SmolVLM2ForConditionalGeneration(config)
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
    print(f"[smolvlm2:weights] 체크포인트에서 로드된 텐서 키: {loaded}/{total} "
          f"(체크포인트 제공 {len(state)}개)")

    hard_missing = [m for m in missing if m not in allowed_missing]
    if hard_missing:
        raise RuntimeError(
            f"[smolvlm2:weights] 누락 키 {len(hard_missing)}개(해당 레이어가 랜덤 초기화됨): "
            f"{hard_missing[:8]} ... config 와 체크포인트가 일치하는지 확인하세요."
        )
    if unexpected:
        raise RuntimeError(
            f"[smolvlm2:weights] 예상 밖 키 {len(unexpected)}개: {unexpected[:8]} ... "
            f"체크포인트 포맷/디렉토리를 확인하세요."
        )
