"""SmolVLM 비전 인코더 (SigLIP, 스크래치 구현).

CLS 토큰 없이 패치 임베딩 + 학습형 위치 임베딩을 쓰는 pre-norm ViT.
모듈/파라미터 이름은 HF SmolVLM(Idefics3) 체크포인트 키와 일치시켜 두었다:
    model.vision_model.embeddings.patch_embedding / position_embedding
    model.vision_model.encoder.layers.N.self_attn.{q,k,v,out}_proj
    model.vision_model.encoder.layers.N.{layer_norm1,layer_norm2,mlp.fc1,mlp.fc2}
    model.vision_model.post_layernorm

이 프레임워크의 프로세서는 각 타일을 정확히 image_size 정사각형으로 리사이즈하므로
패치가 전부 유효하다 → patch_attention_mask 없이 완전 양방향 어텐션을 쓴다(Flash 가능).

텐서 차원 표기:
    B = (배치×이미지) = 실제 타일 수, C = 채널(3), p = patch_size
    N = num_patches = (image_size/p)², D = hidden_size
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import SmolVLMVisionConfig


# ------------------------------------------------------------------ #
# full grid(모든 패치 유효, 한 변 n 패치)에서의 위치 id (1, n²) 를 만든다.
# HF Idefics3/SmolVLM 의 bucketize 스킴을 그대로 따른다:
#   fractional = (i/n)·(1-1e-6),  boundaries = [1/n, 2/n, ..., (n-1)/n]
#   pos = bucketize(fractional, boundaries, right=True)  (h,w 각각) → h·n + w
# (n=num_patches_per_side 이므로 결과는 arange 와 다르다 — 경계값 근처에서 한 칸 밀린다.)
# ------------------------------------------------------------------ #
def _full_grid_position_ids(n: int) -> torch.Tensor:
    boundaries = torch.arange(1 / n, 1.0, 1 / n)
    idx        = torch.arange(n, dtype=torch.float32) / n * (1 - 1e-6)
    bucket     = torch.bucketize(idx, boundaries, right=True)          # (n,)
    pos_ids    = (bucket[:, None] * n + bucket[None, :]).flatten()     # (n²,)
    return pos_ids.long().unsqueeze(0)


class SmolVLMVisionEmbeddings(nn.Module):
    """이미지를 패치 임베딩 시퀀스로 변환하고 위치 임베딩을 더한다."""

    # ------------------------------------------------------------------ #
    # 패치 임베딩(Conv2d)과 위치 임베딩(nn.Embedding)을 만든다.
    # Conv2d 는 kernel=stride=patch_size 라 겹치지 않는 패치별 선형투영과 동일하다.
    #
    # ★ 위치 id 는 단순 arange 가 아니라 HF Idefics3/SmolVLM 의 bucketize 스킴으로
    #   계산한다(가변해상도용 NaFlex 방식). 위치 임베딩 테이블이 이 스킴으로 학습됐기
    #   때문에, 사전학습 가중치를 정확히 쓰려면 동일하게 재현해야 한다. 이 프레임워크는
    #   타일을 항상 image_size 정사각(모든 패치 유효)으로 넣으므로 id 가 상수 → 버퍼로 캐시.
    # ------------------------------------------------------------------ #
    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",
        )
        self.num_positions      = config.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids", _full_grid_position_ids(config.num_patches_per_side), persistent=False
        )

    # ------------------------------------------------------------------ #
    # (B,C,H,W) 이미지 → (B, N, D) 패치 임베딩 시퀀스(+위치 임베딩).
    # ------------------------------------------------------------------ #
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)          # (B, D, H/p, W/p)
        embeddings   = patch_embeds.flatten(2).transpose(1, 2)     # (B, N, D)
        return embeddings + self.position_embedding(self.position_ids)


class SmolVLMVisionAttention(nn.Module):
    """표준 멀티헤드 셀프 어텐션 (마스크 없음, 완전 양방향)."""

    # ------------------------------------------------------------------ #
    # Q/K/V/출력 투영과 헤드 분할 파라미터를 준비한다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim  = self.embed_dim // self.num_heads
        self.dropout   = config.attention_dropout

        self.q_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # ------------------------------------------------------------------ #
    # 패치 전체에 대한 양방향 셀프 어텐션. 마스크가 없어 SDPA 가 Flash 커널을 쓸 수 있다.
    # ------------------------------------------------------------------ #
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, n, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(b, n, self.embed_dim)
        return self.out_proj(out)


class SmolVLMVisionMLP(nn.Module):
    """두 개의 선형층 + gelu(tanh 근사) 피드포워드."""

    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = nn.functional.gelu(x, approximate="tanh")
        return self.fc2(x)


class SmolVLMVisionEncoderLayer(nn.Module):
    """Pre-norm 잔차 블록 (LayerNorm → attn/mlp → 잔차)."""

    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn   = SmolVLMVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp         = SmolVLMVisionMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class SmolVLMVisionEncoder(nn.Module):
    """인코더 레이어를 num_hidden_layers 개 쌓은 스택 (gradient checkpointing 지원)."""

    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SmolVLMVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class SmolVLMVisionTransformer(nn.Module):
    """임베딩 → 인코더 스택 → 최종 LayerNorm 으로 이어지는 ViT 본체.

    (HF 키 model.vision_model.* 에 대응 — 최상위 이름이 곧 vision_model 이다.)
    """

    def __init__(self, config: SmolVLMVisionConfig):
        super().__init__()
        self.config         = config
        self.embeddings     = SmolVLMVisionEmbeddings(config)
        self.encoder        = SmolVLMVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # ------------------------------------------------------------------ #
    # 이미지 → 임베딩 → 인코더 → post_layernorm.
    # 풀링 없이 패치 시퀀스 (B, N, D) 를 그대로 반환한다.
    # patch_attention_mask 는 인터페이스 호환용이며(전 타일 유효) 사용하지 않는다.
    # ------------------------------------------------------------------ #
    def forward(self, pixel_values: torch.Tensor, patch_attention_mask=None) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        return self.post_layernorm(hidden_states)                  # (B, N, D)
