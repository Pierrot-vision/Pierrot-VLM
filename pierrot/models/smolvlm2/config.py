"""SmolVLM2 모델 설정 정의.

SmolVLM2(= SigLIP 비전 인코더 + 픽셀셔플 커넥터 + SmolLM2(Llama 계열) 언어 모델)
구조를 스크래치로 재현하기 위한 dataclass 설정들이다. SmolVLM 은 HuggingFace
Idefics3 아키텍처를 그대로 따르므로, 모듈/파라미터 이름을 HF 체크포인트 키와
1:1 로 맞춰 두면 `load_state_dict(strict=False)` 로 공개 가중치가 바로 로드된다.

★ 필드 이름은 SmolVLM2 공개 체크포인트의 config.json 과 맞춰져 있다. ★
  → 공식 체크포인트는 크기마다 다른 키를 생략하고 HF 기본값에 의존하므로,
    weights.config_from_json() 은 HF AutoConfig 로 기본값까지 해석해 이 dataclass 로 매핑한다
    (256M/500M/2.2B 모두 정확히 로드됨). 아래 dataclass 기본값은 pretrained=None 스크래치
    학습에서만 쓰이며, `HuggingFaceTB/SmolVLM2-2.2B-Instruct` 구조 기준의 "안전한 기본값"이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SmolVLMVisionConfig:
    """SigLIP 비전 인코더 설정 (SmolVLM 의 vision_config)."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 384          # 타일 한 변 크기(=분할 단위). 256M/500M 은 512.
    patch_size: int = 14           # 256M/500M 은 16.
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    # ------------------------------------------------------------------ #
    # 한 변당 패치 수 = image_size // patch_size (conv stride=patch, 패딩 없음).
    # ------------------------------------------------------------------ #
    @property
    def num_patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    # ------------------------------------------------------------------ #
    # 타일 하나의 패치 개수(=위치 임베딩 개수) = (image_size // patch_size)².
    # ------------------------------------------------------------------ #
    @property
    def num_patches(self) -> int:
        return self.num_patches_per_side ** 2


@dataclass
class SmolLM2TextConfig:
    """SmolLM2 / Llama 계열 언어 모델 설정 (SmolVLM 의 text_config)."""

    # 기본값은 공식 SmolVLM2-2.2B 의 text_config(= SmolLM2-1.7B, Llama·MHA) 기준이다.
    # (공식 config.json 이 num_attention_heads 등을 생략하고 HF 클래스 기본값에 의존하므로,
    #  여기 기본값도 그 값과 일치시켜 둔다. pretrained 경로는 config.json 값이 우선.)
    vocab_size: int = 49280
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 32       # 공식 2.2B: MHA(=heads). 소형(256M/500M)은 GQA.
    head_dim: Optional[int] = None      # None 이면 hidden_size // num_attention_heads (=64)
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 130000.0
    attention_dropout: float = 0.0
    pad_token_id: Optional[int] = None

    # ------------------------------------------------------------------ #
    # head_dim 미지정 시 hidden_size / num_attention_heads 로 채운다.
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    # ------------------------------------------------------------------ #
    # GQA 그룹 수 = Q 헤드 수 / KV 헤드 수 (KV 헤드당 공유하는 Q 헤드 수).
    # ------------------------------------------------------------------ #
    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass
class SmolVLM2Config:
    """최상위 SmolVLM2 설정 (비전+언어 설정을 묶고 커넥터/이미지 토큰을 배선)."""

    text_config: SmolLM2TextConfig = field(default_factory=SmolLM2TextConfig)
    vision_config: SmolVLMVisionConfig = field(default_factory=SmolVLMVisionConfig)

    # 픽셀셔플 축소 배율 s: 비전 패치 시퀀스를 s² 배로 압축한다.
    #   타일당 이미지 토큰 수 = (image_size//patch_size)² // s².
    #   2.2B: 729//9=81, 256M/500M: 1024//16=64.
    scale_factor: int = 3
    image_token_id: int = 49190     # <image> placeholder 토큰 id (프로세서가 확정)
    ignore_index: int = -100
    vocab_size: int = 49280         # = text_config.vocab_size
    tie_word_embeddings: bool = False
    # ★ 최상위 pad_token_id 는 '멀티모달/프로세서용' 메타데이터다(공식 2.2B=128002).
    #   이는 text_config.pad_token_id(공식=2)와 다르며, vocab(49280) 밖의 값일 수 있다.
    #   따라서 이 값을 절대 text_config.pad_token_id(=텍스트 임베딩 padding_idx)로
    #   전파하지 않는다. 전파하면 nn.Embedding(vocab, ..., padding_idx=128002) 가
    #   "padding_idx must be within num_embeddings" 로 즉시 실패한다.
    pad_token_id: Optional[int] = None

    # ------------------------------------------------------------------ #
    # 하위 config 를 정돈하고 교차 배선을 맞춘다.
    #   - dict 로 들어온 text/vision config 를 dataclass 로 승격(config.json 대응)
    #   - 최상위 vocab 을 언어 config 와 동기화 (pad 는 전파하지 않음 — 위 주석 참고)
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if isinstance(self.text_config, dict):
            self.text_config = SmolLM2TextConfig(**_filter(SmolLM2TextConfig, self.text_config))
        if isinstance(self.vision_config, dict):
            self.vision_config = SmolVLMVisionConfig(**_filter(SmolVLMVisionConfig, self.vision_config))

        self.vocab_size = self.text_config.vocab_size

    # ------------------------------------------------------------------ #
    # 타일당 이미지 토큰 수(= <image> placeholder 개수) = 패치수 // scale_factor².
    # 프로세서·병합 로직이 참조하는 핵심 값(inputs_merger 의 블록 크기 S).
    # ------------------------------------------------------------------ #
    @property
    def image_seq_len(self) -> int:
        return self.vision_config.num_patches // (self.scale_factor ** 2)


# ------------------------------------------------------------------ #
# dataclass 필드에 존재하는 키만 남긴다 (HF config.json 의 여분 키 무시).
# ------------------------------------------------------------------ #
def _filter(cls, d: dict) -> dict:
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in d.items() if k in valid}
