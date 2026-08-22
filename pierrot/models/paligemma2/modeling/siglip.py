"""SigLIP-So400m 비전 인코더 (스크래치 구현).

CLS 토큰 없이 패치 임베딩 + 학습형 위치 임베딩을 사용하는 pre-norm ViT.
모듈/파라미터 이름은 HuggingFace 체크포인트 키와 일치시켜 `load_state_dict` 가
바로 되도록 했다 (vision_tower.vision_model.encoder.layers.N.self_attn.q_proj ...).

텐서 차원 표기:
    B = 배치, C = 채널(3), H/W = 이미지 크기(896), p = patch_size(14)
    N = num_patches = (896/14)² = 4096, D = hidden_size(1152)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import SiglipConfig


class SiglipVisionEmbeddings(nn.Module):
    """이미지를 패치 임베딩 시퀀스로 변환하고 위치 임베딩을 더한다."""

    # ------------------------------------------------------------------ #
    # 패치 임베딩(Conv2d)과 위치 임베딩(nn.Embedding)을 만든다.
    # Conv2d 는 kernel=stride=patch_size 라 겹치지 않는 패치별 선형투영과 동일하다.
    # 위치 인덱스(position_ids)는 학습되지 않는 버퍼로 등록한다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config    = config
        self.embed_dim = config.hidden_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",  # 패딩 없음
        )
        self.num_patches        = config.num_patches
        self.num_positions      = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    # (B,C,H,W) 이미지 → (B, num_patches, D) 패치 임베딩 시퀀스.
    # Conv2d 로 패치화 후 공간축을 펼쳐 전치하고, 위치 임베딩을 더한다.
    # ------------------------------------------------------------------ #
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B, C, H, W)
        patch_embeds = self.patch_embedding(pixel_values)          # (B, D, H/p, W/p)
        embeddings   = patch_embeds.flatten(2).transpose(1, 2)       # (B, num_patches, D)
        embeddings   = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    """표준 멀티헤드 셀프 어텐션 (마스크 없음, 완전 양방향)."""

    # ------------------------------------------------------------------ #
    # Q/K/V/출력 투영과 헤드 분할 파라미터를 준비한다.
    # scale = 1/sqrt(head_dim) (표준 스케일드 닷프로덕트).
    # ------------------------------------------------------------------ #
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config    = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim  = self.embed_dim // self.num_heads
        self.scale     = self.head_dim ** -0.5
        self.dropout   = config.attention_dropout

        self.q_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # ------------------------------------------------------------------ #
    # 패치 전체에 대한 양방향 셀프 어텐션(마스크 없음).
    # 마스크가 없어 SDPA 가 최신 GPU 에서 실제 Flash 커널을 쓸 수 있다. 896(4096 패치)
    # 에서 head 별 L×L score 저장을 피해 vision 학습(unfreeze) 시 메모리를 크게 아낀다.
    # ------------------------------------------------------------------ #
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, n, _ = hidden_states.shape

        # (B, heads, N, head_dim)
        q = self.q_proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)  # 마스크 없음 → Flash 가능

        out = out.transpose(1, 2).reshape(b, n, self.embed_dim)
        return self.out_proj(out)


class SiglipMLP(nn.Module):
    """두 개의 선형층 + gelu(tanh 근사) 피드포워드."""

    # ------------------------------------------------------------------ #
    # 확장(fc1: hidden→intermediate)·축소(fc2: intermediate→hidden) 선형층 생성.
    # ------------------------------------------------------------------ #
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    # ------------------------------------------------------------------ #
    # fc1 → gelu(tanh 근사) → fc2 순으로 통과시킨다.
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:(B,N,D) → fc1:(B,N,intermediate) → gelu → fc2:(B,N,D)
        x = self.fc1(x)
        x = nn.functional.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class SiglipEncoderLayer(nn.Module):
    """Pre-norm 잔차 블록 (LayerNorm → attn/mlp → 잔차)."""

    # ------------------------------------------------------------------ #
    # 어텐션/MLP 서브블록과 각 앞단 LayerNorm 두 개를 만든다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn   = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp         = SiglipMLP(config)

    # ------------------------------------------------------------------ #
    # pre-norm 잔차: x += attn(ln1(x)); x += mlp(ln2(x)).
    # ------------------------------------------------------------------ #
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 전 구간 (B, N, D) 모양 불변. pre-norm 후 잔차로 더함.
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class SiglipEncoder(nn.Module):
    """인코더 레이어를 num_hidden_layers 개 쌓은 스택."""

    # ------------------------------------------------------------------ #
    # SiglipEncoderLayer 를 num_hidden_layers 개 담은 ModuleList 를 만든다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    # ------------------------------------------------------------------ #
    # 모든 레이어를 순차 통과시킨다.
    # ------------------------------------------------------------------ #
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 27개 레이어 순차 통과, (B, N, D) 모양 불변
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """임베딩 → 인코더 스택 → 최종 LayerNorm 으로 이어지는 ViT 본체."""

    # ------------------------------------------------------------------ #
    # 패치/위치 임베딩, 인코더 스택, 최종 post_layernorm 을 조립한다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.embeddings     = SiglipVisionEmbeddings(config)
        self.encoder        = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # ------------------------------------------------------------------ #
    # 이미지 → 임베딩 → 인코더 → post_layernorm.
    # 풀링 없이 패치 시퀀스 (B, num_patches, D) 를 그대로 반환한다.
    # ------------------------------------------------------------------ #
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # (B,3,896,896) → 패치임베딩 (B,N,D) → 인코더 (B,N,D) → norm (B,N,D)
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states                                       # (B, num_patches, D)


class SiglipVisionModel(nn.Module):
    """비전 타워 최상위 래퍼 (HF 키 vision_tower.vision_model.* 에 대응)."""

    # ------------------------------------------------------------------ #
    # 내부 ViT 본체(vision_model)를 생성한다. 이름을 HF 키와 맞춘다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config       = config
        self.vision_model = SiglipVisionTransformer(config)

    # ------------------------------------------------------------------ #
    # pixel_values 를 내부 ViT 로 인코딩해 패치 특징을 반환한다.
    # ------------------------------------------------------------------ #
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)
