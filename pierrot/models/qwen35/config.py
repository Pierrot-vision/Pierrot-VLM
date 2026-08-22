"""Qwen3.5 모델 설정 정의.

Qwen3.5(= 동적 해상도 ViT + 패치 머저 + **하이브리드 디코더**) 구조를 스크래치로
재현하기 위한 dataclass 설정들이다. 모듈/파라미터 이름을 HF
`Qwen3_5ForConditionalGeneration` 체크포인트 키와 1:1 로 맞춰 두면
`load_state_dict(strict=False)` 로 공개 가중치가 바로 로드된다.

★ 필드 이름은 Qwen3.5 공개 체크포인트의 config.json 과 맞춰져 있다. ★
  → Qwen3.5 의 config.json 도 모든 구조 키를 명시하므로 HF AutoConfig 없이 raw JSON
    파싱만으로 정확히 복원된다. 덕분에 Qwen3.5 를 아직 모르는 transformers(<5.1)
    환경에서도 이 구현은 그대로 동작한다.
    아래 dataclass 기본값은 `Qwen/Qwen3.5-4B` 기준이다.

Qwen3.5 가 Qwen3-VL 과 결정적으로 다른 점(비전/전처리는 동일 계약, DeepStack 없음):
  - 디코더가 **하이브리드**다. 32개 레이어 중 4번째마다 full attention, 나머지는
    **Gated DeltaNet**(linear attention, 상태 기반 recurrence)이다 → layer_types.
  - full attention 은 **출력 게이트**(attn_output_gate)를 가진다: q_proj 가 쿼리와
    게이트를 함께 뽑고, 어텐션 출력에 sigmoid(gate) 를 곱한 뒤 o_proj 를 지난다.
  - RoPE 가 **부분 회전**(partial_rotary_factor=0.25)이다: head_dim 256 중 앞 64
    차원만 회전하고 나머지는 그대로 둔다. M-RoPE 배분(mrope_section=[11,11,10])은
    이 회전 차원(64/2=32개 주파수) 안에서 이뤄진다.
  - RMSNorm 이 **zero-centered** 다: 가중치를 0 근처로 저장하고 (1 + weight) 로
    곱한다(가중치 감쇠와 궁합). 체크포인트 값이 이 규약이라 로드 시 반드시 맞춰야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Qwen35VisionConfig:
    """Qwen3.5 비전 인코더 설정 (config.json 의 vision_config, DeepStack 없음)."""

    hidden_size: int = 1024
    intermediate_size: int = 4096
    depth: int = 24                        # 비전 블록 수
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16                   # 공간 패치 한 변
    temporal_patch_size: int = 2           # 시간 패치(정지 이미지는 같은 프레임 2회 복제)
    spatial_merge_size: int = 2            # 패치 머저 압축 배율 m (이미지 토큰 = 패치수/m²)
    num_position_embeddings: int = 2304    # 학습형 위치 임베딩 격자 = 48×48
    out_hidden_size: int = 2560            # 머저 출력 차원(= 언어 hidden_size)
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6

    # ------------------------------------------------------------------ #
    # 어텐션 헤드 하나의 차원 = hidden_size // num_heads.
    # ------------------------------------------------------------------ #
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    # ------------------------------------------------------------------ #
    # 학습형 위치 임베딩 격자의 한 변(= √num_position_embeddings, 공식 48).
    # ------------------------------------------------------------------ #
    @property
    def num_grid_per_side(self) -> int:
        return int(self.num_position_embeddings ** 0.5)

    # ------------------------------------------------------------------ #
    # 패치 하나를 펼친 벡터의 길이 = C × T × p × p (프로세서 출력 마지막 차원).
    # ------------------------------------------------------------------ #
    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size ** 2


@dataclass
class Qwen35TextConfig:
    """Qwen3.5 하이브리드 언어 모델 설정 (config.json 의 text_config)."""

    vocab_size: int = 248320
    hidden_size: int = 2560
    intermediate_size: int = 9216
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4           # GQA (Q 16 : KV 4 = 4배 공유)
    head_dim: Optional[int] = 256          # hidden/heads(=160) 와 무관하게 256 고정
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    pad_token_id: Optional[int] = None
    # ── M-RoPE + 부분 회전 (Qwen3.5 고유 조합) ──
    # head_dim(256)×0.25 = 64 차원만 회전. 그 32개 주파수를 [11,11,10]으로 배분.
    partial_rotary_factor: float = 0.25
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    mrope_interleaved: bool = True
    # ── 하이브리드 레이어 배치 ──
    # layer_types 를 명시하지 않으면 full_attention_interval 번째마다 full attention.
    # (공식 4B: 3,7,11,...,31 이 full — 즉 (i+1) % 4 == 0)
    full_attention_interval: int = 4
    layer_types: Optional[List[str]] = None
    # ── Gated DeltaNet (linear attention 레이어) ──
    linear_conv_kernel_dim: int = 4        # 인과 depthwise conv 커널 크기
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32       # V 헤드가 K 헤드의 2배 → q/k 를 2배 복제
    hidden_act: str = "silu"

    # ------------------------------------------------------------------ #
    # head_dim 미지정 시 hidden/heads 로 채우고, layer_types 미지정 시
    # full_attention_interval 로 하이브리드 배치를 유도한다.
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types 길이({len(self.layer_types)})가 num_hidden_layers"
                f"({self.num_hidden_layers})와 일치해야 합니다."
            )

    # ------------------------------------------------------------------ #
    # GQA 그룹 수 = Q 헤드 수 / KV 헤드 수.
    # ------------------------------------------------------------------ #
    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    # ------------------------------------------------------------------ #
    # RoPE 가 실제로 회전시키는 차원 수 = head_dim × partial_rotary_factor.
    # ------------------------------------------------------------------ #
    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    # ------------------------------------------------------------------ #
    # Gated DeltaNet 의 키/값 전체 차원과 conv 채널 수.
    # ------------------------------------------------------------------ #
    @property
    def linear_key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def linear_value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def linear_conv_dim(self) -> int:
        return self.linear_key_dim * 2 + self.linear_value_dim


@dataclass
class Qwen35Config:
    """최상위 Qwen3.5 설정 (비전+언어 설정을 묶고 특수 토큰을 배선)."""

    text_config: Qwen35TextConfig = field(default_factory=Qwen35TextConfig)
    vision_config: Qwen35VisionConfig = field(default_factory=Qwen35VisionConfig)

    image_token_id: int = 248056           # <|image_pad|> (이미지 placeholder)
    video_token_id: int = 248057           # <|video_pad|> (이 구현은 이미지 전용 — 예약)
    vision_start_token_id: int = 248053    # <|vision_start|>
    vision_end_token_id: int = 248054      # <|vision_end|>
    ignore_index: int = -100
    vocab_size: int = 248320               # = text_config.vocab_size
    tie_word_embeddings: bool = True       # Qwen3.5-4B 는 lm_head 를 임베딩과 공유

    # ------------------------------------------------------------------ #
    # 하위 config 를 정돈하고 교차 배선을 맞춘다(qwen3vl 과 동일 규약).
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if isinstance(self.text_config, dict):
            self.text_config = Qwen35TextConfig(**_filter(Qwen35TextConfig, self.text_config))
        if isinstance(self.vision_config, dict):
            self.vision_config = Qwen35VisionConfig(**_filter(Qwen35VisionConfig, self.vision_config))

        self.vocab_size = self.text_config.vocab_size
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"vision_config.out_hidden_size({self.vision_config.out_hidden_size}) 와 "
                f"text_config.hidden_size({self.text_config.hidden_size}) 가 달라 이미지 병합이 불가능합니다."
            )

    # ------------------------------------------------------------------ #
    # 격자 (grid_t, grid_h, grid_w) 하나가 만들어 내는 이미지 토큰 수.
    # ------------------------------------------------------------------ #
    def image_tokens_for_grid(self, grid_t: int, grid_h: int, grid_w: int) -> int:
        return grid_t * grid_h * grid_w // (self.vision_config.spatial_merge_size ** 2)


# ------------------------------------------------------------------ #
# dataclass 필드에 존재하는 키만 남긴다 (HF config.json 의 여분 키 무시).
# ------------------------------------------------------------------ #
def _filter(cls, d: dict) -> dict:
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in d.items() if k in valid}
