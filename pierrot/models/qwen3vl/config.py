"""Qwen3-VL 모델 설정 정의.

Qwen3-VL(= 동적 해상도 ViT + DeepStack + 패치 머저 + Qwen3 언어 모델) 구조를
스크래치로 재현하기 위한 dataclass 설정들이다. 모듈/파라미터 이름을 HF
`Qwen3VLForConditionalGeneration` 체크포인트 키와 1:1 로 맞춰 두면
`load_state_dict(strict=False)` 로 공개 가중치가 바로 로드된다.

★ 필드 이름은 Qwen3-VL 공개 체크포인트의 config.json 과 맞춰져 있다. ★
  → Qwen3-VL 의 config.json 은 (SmolVLM 과 달리) 모든 키를 명시하므로 HF AutoConfig
    없이 raw JSON 파싱만으로 정확히 복원된다. 덕분에 transformers 4.57(Qwen3-VL 지원)
    미만 환경에서도 이 구현은 그대로 동작한다.
    아래 dataclass 기본값은 `Qwen/Qwen3-VL-2B-Instruct` 기준이다.

Qwen3-VL 이 다른 세 모델(PaliGemma2/nanoVLM/SmolVLM2)과 결정적으로 다른 점:
  - 이미지를 정사각 타일로 자르지 않고, **원본 종횡비를 유지한 채** 32(=patch×merge)
    배수로 리사이즈해 가변 격자(grid_thw)로 본다 → 이미지 토큰 수가 이미지마다 다르다.
  - 위치 임베딩은 고정 격자(48×48=2304)를 **bilinear 보간**해 임의 격자에 맞춘다.
  - 언어 모델 RoPE 가 3D(**M-RoPE**: 시간/높이/너비)이며 interleaved 배치를 쓴다.
  - 비전 중간층(**DeepStack**) 특징을 언어 디코더 앞쪽 레이어에 더해 넣는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Qwen3VLVisionConfig:
    """Qwen3-VL 비전 인코더 설정 (config.json 의 vision_config)."""

    hidden_size: int = 1024
    intermediate_size: int = 4096
    depth: int = 24                        # 비전 블록 수
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16                   # 공간 패치 한 변
    temporal_patch_size: int = 2           # 시간 패치(정지 이미지는 같은 프레임 2회 복제)
    spatial_merge_size: int = 2            # 패치 머저 압축 배율 m (이미지 토큰 = 패치수/m²)
    num_position_embeddings: int = 2304    # 학습형 위치 임베딩 격자 = 48×48
    out_hidden_size: int = 2048            # 머저 출력 차원(= 언어 hidden_size)
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    # DeepStack: 이 인덱스의 비전 블록 출력을 별도 머저로 뽑아 언어 디코더에 주입한다.
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [5, 11, 17])

    # ------------------------------------------------------------------ #
    # 어텐션 헤드 하나의 차원 = hidden_size // num_heads.
    # ------------------------------------------------------------------ #
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    # ------------------------------------------------------------------ #
    # 학습형 위치 임베딩 격자의 한 변(= √num_position_embeddings, 공식 48).
    # 실제 이미지 격자(h, w)에는 이 격자를 bilinear 보간해 맞춘다.
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
class Qwen3VLTextConfig:
    """Qwen3 언어 모델 설정 (config.json 의 text_config)."""

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8           # GQA (Q 16 : KV 8 = 2배 공유)
    head_dim: Optional[int] = 128          # Qwen3 는 hidden/heads 와 무관하게 128 고정
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    pad_token_id: Optional[int] = None
    # ── M-RoPE (Qwen3-VL 고유) ──
    # head_dim/2(=64)개 주파수를 시간/높이/너비 세 축에 [24,20,20]으로 배분한다.
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    # True 면 [TTT...HHH...WWW] 청크가 아니라 [THWTHW...] 로 교차 배치한다(Qwen3-VL 기본).
    mrope_interleaved: bool = True

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
class Qwen3VLConfig:
    """최상위 Qwen3-VL 설정 (비전+언어 설정을 묶고 특수 토큰을 배선)."""

    text_config: Qwen3VLTextConfig = field(default_factory=Qwen3VLTextConfig)
    vision_config: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)

    image_token_id: int = 151655           # <|image_pad|> (이미지 placeholder)
    video_token_id: int = 151656           # <|video_pad|> (이 구현은 이미지 전용 — 예약)
    vision_start_token_id: int = 151652    # <|vision_start|>
    vision_end_token_id: int = 151653      # <|vision_end|>
    ignore_index: int = -100
    vocab_size: int = 151936               # = text_config.vocab_size
    tie_word_embeddings: bool = True       # Qwen3-VL-2B 는 lm_head 를 임베딩과 공유

    # ------------------------------------------------------------------ #
    # 하위 config 를 정돈하고 교차 배선을 맞춘다.
    #   - dict 로 들어온 text/vision config 를 dataclass 로 승격(config.json 대응)
    #   - 최상위 vocab 을 언어 config 와 동기화
    #   - 비전 머저 출력(out_hidden_size)이 언어 hidden_size 와 다르면 즉시 오류
    #     (다르면 이미지 임베딩을 텍스트 시퀀스에 끼울 수 없다)
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if isinstance(self.text_config, dict):
            self.text_config = Qwen3VLTextConfig(**_filter(Qwen3VLTextConfig, self.text_config))
        if isinstance(self.vision_config, dict):
            self.vision_config = Qwen3VLVisionConfig(**_filter(Qwen3VLVisionConfig, self.vision_config))

        self.vocab_size = self.text_config.vocab_size
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"vision_config.out_hidden_size({self.vision_config.out_hidden_size}) 와 "
                f"text_config.hidden_size({self.text_config.hidden_size}) 가 달라 이미지 병합이 불가능합니다."
            )

    # ------------------------------------------------------------------ #
    # 격자 (grid_t, grid_h, grid_w) 하나가 만들어 내는 이미지 토큰 수.
    #   = t·h·w / m²  (머저가 m×m 이웃 패치를 하나로 합치므로)
    # 프로세서(placeholder 개수)와 모델(병합 검증)이 같은 식을 참조한다.
    # ------------------------------------------------------------------ #
    def image_tokens_for_grid(self, grid_t: int, grid_h: int, grid_w: int) -> int:
        return grid_t * grid_h * grid_w // (self.vision_config.spatial_merge_size ** 2)


# ------------------------------------------------------------------ #
# dataclass 필드에 존재하는 키만 남긴다 (HF config.json 의 여분 키 무시).
# ------------------------------------------------------------------ #
def _filter(cls, d: dict) -> dict:
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in d.items() if k in valid}
