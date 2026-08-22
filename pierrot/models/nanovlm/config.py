"""nanoVLM 모델 설정 정의.

nanoVLM(= SigLIP2 ViT 비전 인코더 + 픽셀셔플 프로젝터 + SmolLM2 계열 언어 디코더)을
Pierrot 규약(스크래치 재현·HF 키 호환)으로 담는 dataclass 설정이다.

★ 필드 이름은 nanoVLM 공개 체크포인트의 config.json 과 1:1 로 맞춰져 있다. ★
  → VLMConfig(**json.load(config.json)) 로 공개 가중치를 그대로 로드할 수 있다.

기본값은 `SmolLM2-360M-Instruct` + `siglip2-base-patch16-512` (≈450M VLM) 기준.
백본에서 시작할 때 vit_*/lm_* 의 상당수는 HF 백본 config 로 로드 시점에 덮어써진다
(weights.py 의 backbone from_pretrained 참고). 즉 여기 값은 "안전한 기본값"이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# ------------------------------------------------------------------ #
# 66개 추가 특수 토큰: <|image|> 플레이스홀더 + <|global_image|> + 8×8 타일 좌표
# (<row_i_col_j>). 이미지 임베딩이 삽입될 자리를 만들고, 토크나이저 속성으로도
# 노출된다(tokenizer.image_token / global_image_token / r{i}c{j}).
# ------------------------------------------------------------------ #
def _default_extra_tokens() -> Dict[str, str]:
    tokens = {
        "image_token": "<|image|>",
        "global_image_token": "<|global_image|>",
    }
    for i in range(1, 9):
        for j in range(1, 9):
            tokens[f"r{i}c{j}"] = f"<row_{i}_col_{j}>"
    return tokens


@dataclass
class VLMConfig:
    """nanoVLM 최상위 설정 (비전 + 언어 + 프로젝터 + 토크나이저)."""

    # ── 비전(SigLIP2 ViT) ──────────────────────────────────────────────
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 3072                  # = 4 × vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 512                    # 타일 한 변 크기(=분할 단위)
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False                 # False = 모든 패치 토큰 사용(풀링 없음)
    vit_model_type: str = "google/siglip2-base-patch16-512"

    # ── 언어(SmolLM2 / Llama 계열 디코더) ──────────────────────────────
    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000                   # RoPE base(theta)
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66               # 추가 특수 토큰 수(=len(vlm_extra_tokens))
    lm_vocab_size: int = 49218                 # = lm_base_vocab_size + extra_token_amount
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5                     # GQA
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 4096
    # ★ VLM 백본으로 쓸 때는 False: 디코더가 토큰이 아니라 임베딩을 입력받고,
    #   lm_head 는 디코더 밖(VLM)에서 적용된다. (이미지 임베딩 병합 경로가 성립)
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True                # head.weight 를 token_embedding 과 공유
    lm_model_type: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_chat_template: str = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )

    # ── 멀티모달 프로젝터(픽셀 셔플) ────────────────────────────────────
    mp_pixel_shuffle_factor: int = 4           # 패치 토큰을 s² 배로 압축(1024→64)
    mp_image_token_length: int = 64            # 타일당 이미지 토큰 수(=(vit_img/patch)²/s²)

    # ── 이미지 전처리 ──────────────────────────────────────────────────
    max_img_size: int = 2048                   # 긴 변 상한
    resize_to_max_side_len: bool = True

    # ── 특수 토큰 / 기타 ───────────────────────────────────────────────
    vlm_extra_tokens: Dict[str, str] = field(default_factory=_default_extra_tokens)
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = "checkpoints"
    hf_repo_name: str = "nanoVLM"
