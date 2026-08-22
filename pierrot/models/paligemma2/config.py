"""PaliGemma2 모델 설정 정의.

PaliGemma2(=SigLIP-So400m 비전 인코더 + Gemma2 언어 모델 + 선형 프로젝터) 구조를
스크래치로 재현하기 위한 dataclass 설정들이다.

기본값은 `google/paligemma2-3b-pt-224` 체크포인트에 맞춰져 있으나,
실제 학습/추론 시에는 체크포인트의 `config.json` 값으로 덮어써진다
(weights.py 참고). 즉 여기 값은 "안전한 기본값"일 뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SiglipConfig:
    """SigLIP-So400m/14 비전 인코더 설정."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 14
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    # 언어 모델 hidden_size와 맞춰지는 프로젝션 차원(프로젝터 출력).
    projection_dim: int = 2304

    # ------------------------------------------------------------------ #
    # 이미지 패치 개수(=이미지 토큰 수) = (image_size // patch_size)².
    # ------------------------------------------------------------------ #
    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    # ------------------------------------------------------------------ #
    # 위치 임베딩 개수(패치 수와 동일, CLS 토큰 없음).
    # ------------------------------------------------------------------ #
    @property
    def num_positions(self) -> int:
        return self.num_patches


@dataclass
class Gemma2Config:
    """Gemma2-2B 언어 모델 설정.

    Gemma1 대비 핵심 차이:
      - 레이어마다 4개의 RMSNorm (pre/post attention, pre/post feedforward)
      - local(sliding window)/global attention 교차
      - attention/final logit soft-capping
      - query_pre_attn_scalar 로 attention 스케일링
    """

    vocab_size: int = 257152
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4          # GQA
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    # attention logit 스케일 = 1/sqrt(query_pre_attn_scalar). 2B에서는 256(=head_dim).
    query_pre_attn_scalar: float = 256.0
    sliding_window: int = 4096
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0
    pad_token_id: Optional[int] = 0
    # 짝수 레이어=local(sliding), 홀수 레이어=global 을 만드는 패턴.
    attn_types: List[str] = field(default_factory=list)
    # 프로세서가 채워주는 값(=(image_size//patch_size)**2).
    num_image_tokens: Optional[int] = None

    # ------------------------------------------------------------------ #
    # attn_types 미지정 시 레이어별 유형을 채운다:
    # 짝수 = local_sliding(슬라이딩 윈도우), 홀수 = global.
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if not self.attn_types:
            # 0,2,4,... = local_sliding / 1,3,5,... = global
            self.attn_types = [
                "local_sliding" if i % 2 == 0 else "global"
                for i in range(self.num_hidden_layers)
            ]

    # ------------------------------------------------------------------ #
    # GQA 그룹 수 = Q 헤드 수 / KV 헤드 수 (KV 헤드당 몇 개의 Q 가 공유하는가).
    # ------------------------------------------------------------------ #
    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass
class PaliGemma2Config:
    """최상위 PaliGemma2 설정 (비전+언어 설정을 묶는다)."""

    text_config: Gemma2Config = field(default_factory=Gemma2Config)
    vision_config: SiglipConfig = field(default_factory=SiglipConfig)

    ignore_index: int           = -100
    image_token_index: int      = 257152       # <image> placeholder 토큰 id (프로세서가 확정)
    vocab_size: int             = 257152
    projection_dim: int         = 2304         # = text hidden_size
    hidden_size: int            = 2304         # = text hidden_size
    pad_token_id: Optional[int] = 0

    # ------------------------------------------------------------------ #
    # 하위 config 를 정돈하고 교차 배선을 맞춘다.
    #   - dict 로 들어온 text/vision config 를 dataclass 로 승격(config.json 대응)
    #   - 프로젝터 출력 차원 = 언어 hidden, 이미지 토큰 수 = 패치 수,
    #     pad/vocab 을 언어 config 와 동기화
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        # dict 로 들어오면 dataclass 로 승격 (config.json 로드 대응).
        if isinstance(self.text_config, dict):
            self.text_config = Gemma2Config(**_filter(Gemma2Config, self.text_config))
        if isinstance(self.vision_config, dict):
            self.vision_config = SiglipConfig(**_filter(SiglipConfig, self.vision_config))

        # 교차 배선. 언어 hidden 이 '유일 진실' — 최상위 hidden_size/projection_dim 은
        # 체크포인트 JSON 값과 어긋날 수 있으므로 text_config.hidden_size 로 강제 동기화한다.
        # (이미지 feature scale 1/sqrt(hidden) 이 Gemma2 의 ×sqrt(hidden) 과 정확히 상쇄되게.)
        self.hidden_size                  = self.text_config.hidden_size
        self.projection_dim               = self.text_config.hidden_size
        self.vision_config.projection_dim = self.text_config.hidden_size
        self.text_config.num_image_tokens = self.vision_config.num_patches
        self.text_config.pad_token_id     = self.pad_token_id
        self.vocab_size                   = self.text_config.vocab_size


# ------------------------------------------------------------------ #
# dataclass 필드에 존재하는 키만 남긴다 (HF config.json 의 여분 키 무시).
# ------------------------------------------------------------------ #
def _filter(cls, d: dict) -> dict:
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in d.items() if k in valid}
