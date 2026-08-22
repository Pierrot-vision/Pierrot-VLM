"""Qwen3-VL 비전 인코더 (동적 해상도 ViT + DeepStack, 스크래치 구현).

다른 세 모델과 달리 이미지를 정사각 타일로 자르지 않는다. 프로세서가 원본 종횡비를
유지한 채 32(=patch_size×spatial_merge_size) 배수로 리사이즈해 만든 **가변 격자**
(grid_thw = 프레임수 T, 세로 패치수 H, 가로 패치수 W)를 그대로 받는다. 따라서
입력은 배치 축이 없는 **패킹된 패치 시퀀스** (총패치수, patch_dim) 이며, 어텐션은
이미지 경계(cu_seqlens)마다 끊어서 각 이미지 안에서만 양방향으로 수행된다.

모듈/파라미터 이름은 HF Qwen3-VL 체크포인트 키(model.visual.*)와 일치시켜 두었다:
    model.visual.patch_embed.proj                 (Conv3d)
    model.visual.pos_embed                        (nn.Embedding, 48×48 격자)
    model.visual.blocks.N.{norm1,norm2}
    model.visual.blocks.N.attn.{qkv,proj}
    model.visual.blocks.N.mlp.{linear_fc1,linear_fc2}
    model.visual.merger.{norm,linear_fc1,linear_fc2}
    model.visual.deepstack_merger_list.N.{norm,linear_fc1,linear_fc2}

텐서 차원 표기:
    S      = 총 패치 수 (배치 내 모든 이미지의 T·H·W 합)
    D      = vision hidden_size, hd = head_dim, m = spatial_merge_size
    S/m²   = 이미지 토큰 수 (머저 출력 길이), Dout = out_hidden_size(= 언어 hidden)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Qwen3VLVisionConfig


# ------------------------------------------------------------------ #
# 격자별 패치 시퀀스 순서를 만드는 재정렬 인덱스.
#
# 프로세서가 내보내는 패치 순서는 단순 래스터(행 우선)가 아니라 **머저 블록 우선**이다:
# m×m 이웃 패치 4개가 연속으로 오고, 그 블록들이 행 우선으로 나열된다. 머저가
# reshape(-1, D·m²) 한 번으로 이웃 패치를 합칠 수 있게 하기 위한 배치다.
# 이 함수는 "래스터 인덱스 → 실제 시퀀스 위치" 매핑을 돌려준다(위치·보간 계산 공용).
# ------------------------------------------------------------------ #
def _merge_block_order(h: int, w: int, m: int, device) -> torch.Tensor:
    h_idx = torch.arange(h, device=device).view(h // m, m)
    w_idx = torch.arange(w, device=device).view(w // m, m)
    return (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten()


# ------------------------------------------------------------------ #
# 2D RoPE 용 (h, w) 위치 인덱스를 만든다 → (S, 2).
# 머저 블록 순서를 따르므로 어텐션에서의 패치 순서와 정확히 일치한다.
# 정지 이미지(T=1)면 그대로, 영상(T>1)이면 같은 공간 인덱스를 T번 반복한다.
# ------------------------------------------------------------------ #
def vision_position_ids(grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
    device   = grid_thw.device
    position = []
    for t, h, w in grid_thw.tolist():
        hpos, wpos = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        block = (h // merge_size, merge_size, w // merge_size, merge_size)
        hpos  = hpos.reshape(block).transpose(1, 2).flatten()
        wpos  = wpos.reshape(block).transpose(1, 2).flatten()
        position.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(position, dim=0)


# ------------------------------------------------------------------ #
# 고정 48×48 위치 임베딩 테이블을 임의 격자 (h, w) 로 **bilinear 보간**하기 위한
# 네 꼭짓점 인덱스와 가중치를 만든다 → (4, S), (4, S).
#   h 축: linspace(0, side-1, h) 로 목표 격자 좌표를 잡고 floor/ceil 두 이웃과
#   그 소수부(frac)로 가중치를 구성한다(w 축 동일). 네 조합이 bilinear 4-tap.
# 반환 인덱스는 pos_embed(nn.Embedding) 에 그대로 넣는 1D 인덱스(h·side + w)다.
# ------------------------------------------------------------------ #
def bilinear_pos_embed_index(
    grid_thw: torch.Tensor, side: int, merge_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = grid_thw.device
    idx_parts:    List[List[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: List[List[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor, w_floor = h_grid.int(), w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac, w_frac = h_grid - h_floor, w_grid - w_floor

        hf_off, hc_off = h_floor * side, h_ceil * side
        corners = [
            (hf_off[:, None] + w_floor[None, :]).flatten(),
            (hf_off[:, None] + w_ceil[None, :]).flatten(),
            (hc_off[:, None] + w_floor[None, :]).flatten(),
            (hc_off[:, None] + w_ceil[None, :]).flatten(),
        ]
        weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]
        reorder = _merge_block_order(h, w, merge_size, device).repeat(t)
        for i in range(4):
            idx_parts[i].append(corners[i][reorder])
            weight_parts[i].append(weights[i][reorder])

    indices = torch.stack([torch.cat(p) for p in idx_parts])
    weights = torch.stack([torch.cat(p) for p in weight_parts])
    return indices, weights


# ------------------------------------------------------------------ #
# 이미지(=어텐션 청크)별 패치 길이 목록을 만든다. 각 프레임이 하나의 청크이며
# 길이는 h·w — 어텐션이 이미지/프레임 경계를 넘지 않게 한다.
# ------------------------------------------------------------------ #
def vision_seq_lengths(grid_thw: torch.Tensor) -> List[int]:
    lengths: List[int] = []
    for t, h, w in grid_thw.tolist():
        lengths.extend([h * w] * t)
    return lengths


class Qwen3VLVisionPatchEmbed(nn.Module):
    """펼쳐진 패치 벡터를 Conv3d(=패치별 선형투영)로 임베딩한다."""

    # ------------------------------------------------------------------ #
    # kernel=stride=(T,p,p) 인 Conv3d 를 만든다. 입력이 이미 패치 단위로 잘려 있어
    # 실제로는 patch_dim → hidden_size 선형변환과 동일하지만, 공개 가중치 shape
    # (D, C, T, p, p)를 그대로 쓰기 위해 Conv3d 형태를 유지한다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.in_channels         = config.in_channels
        self.patch_size          = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim           = config.hidden_size

        kernel    = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel, stride=kernel, bias=True)

    # ------------------------------------------------------------------ #
    # (S, patch_dim) → (S, D).
    # ------------------------------------------------------------------ #
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        ).to(self.proj.weight.dtype)
        return self.proj(x).view(-1, self.embed_dim)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """비전 2D RoPE: (S, 2) 위치 인덱스 → (S, hd/2) 각도."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------ #
    # (S, 2) → (S, 2·dim/2). h 축과 w 축 각도를 이어붙여 헤드 차원의 절반을 채운다.
    # ------------------------------------------------------------------ #
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq.to(position_ids.device)).flatten(1)


# ------------------------------------------------------------------ #
# 마지막 차원을 반으로 나눠 뒤 절반을 부호반전해 앞으로 회전(RoPE 보조).
# ------------------------------------------------------------------ #
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# ------------------------------------------------------------------ #
# 비전 q,k 에 RoPE 를 적용한다. (S, heads, hd) 레이아웃이라 cos/sin 은 헤드축
# (dim=-2)으로 브로드캐스트한다. 정밀도 손실을 막기 위해 float32 로 계산한다.
# ------------------------------------------------------------------ #
def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_dtype = q.dtype
    cos, sin   = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    qf, kf     = q.float(), k.float()
    q_embed = (qf * cos) + (rotate_half(qf) * sin)
    k_embed = (kf * cos) + (rotate_half(kf) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class Qwen3VLVisionAttention(nn.Module):
    """이미지 단위 양방향 셀프 어텐션 (융합 qkv, bias 있음)."""

    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim  = config.head_dim
        self.qkv       = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj      = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    # ------------------------------------------------------------------ #
    # (S, D) → qkv 분리·2D RoPE → 이미지별 청크로 나눠 SDPA → (S, D).
    # seq_lengths 로 청크를 나누므로 서로 다른 이미지의 패치가 섞이지 않는다.
    # 마스크를 쓰지 않아(청크마다 완전 양방향) SDPA 가 Flash 커널을 탈 수 있다.
    # ------------------------------------------------------------------ #
    def forward(self, hidden_states: torch.Tensor, seq_lengths: List[int],
                position_embeddings) -> torch.Tensor:
        S = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states).reshape(S, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # (S, heads, hd) → (1, heads, S, hd) 로 바꿔 SDPA 규약에 맞춘다.
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))

        if len(seq_lengths) == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            chunks = [torch.split(t, seq_lengths, dim=2) for t in (q, k, v)]
            out    = torch.cat(
                [F.scaled_dot_product_attention(qc, kc, vc) for qc, kc, vc in zip(*chunks)], dim=2
            )

        out = out.squeeze(0).transpose(0, 1).reshape(S, -1)
        return self.proj(out)


class Qwen3VLVisionMLP(nn.Module):
    """두 선형층 + GELU(tanh 근사) 피드포워드 (bias 있음)."""

    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class Qwen3VLVisionBlock(nn.Module):
    """Pre-norm 잔차 블록 (LayerNorm → attn/mlp → 잔차)."""

    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn  = Qwen3VLVisionAttention(config)
        self.mlp   = Qwen3VLVisionMLP(config)

    def forward(self, hidden_states, seq_lengths, position_embeddings):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), seq_lengths, position_embeddings)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionPatchMerger(nn.Module):
    """m×m 이웃 패치를 하나로 합쳐 언어 hidden 차원으로 투영한다.

    패치 시퀀스가 머저 블록 우선 순서로 정렬돼 있으므로 `view(-1, D·m²)` 한 번이면
    이웃 패치가 채널 방향으로 접힌다(별도 permute 불필요).

    use_postshuffle_norm: 정규화 시점 차이다.
      - False(본 머저)      : 합치기 **전** 패치 차원(D)에서 LayerNorm
      - True(DeepStack 머저): 합친 **후** 확장 차원(D·m²)에서 LayerNorm
    두 경우 norm 파라미터 크기가 달라, 공개 가중치를 쓰려면 반드시 구분해야 한다.
    """

    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size          = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm       = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=config.layer_norm_eps
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    # ------------------------------------------------------------------ #
    # (S, D) → (S/m², Dout).
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x)
        x = x.view(-1, self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3VLVisionModel(nn.Module):
    """Qwen3-VL 비전 타워 본체 (HF 키 model.visual.* 에 대응)."""

    # ------------------------------------------------------------------ #
    # 패치 임베딩 · 학습형 위치 임베딩 · 2D RoPE · 블록 스택 · 머저(+DeepStack 머저)를
    # 구성한다. DeepStack 머저는 deepstack_visual_indexes 개수만큼 만든다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.config             = config
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side  = config.num_grid_per_side

        self.patch_embed  = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed    = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list    = nn.ModuleList([
            Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in self.deepstack_visual_indexes
        ])

    # ------------------------------------------------------------------ #
    # 패킹된 패치 시퀀스를 인코딩한다.
    #   ① 패치 임베딩 → ② 48×48 위치 임베딩을 격자에 bilinear 보간해 덧셈
    #   → ③ 2D RoPE(cos/sin) 준비 → ④ 블록 스택(이미지 경계로 어텐션 분리)
    #   → ⑤ 머저로 m² 압축 (+ DeepStack 인덱스 층에서 중간 특징도 압축해 수집)
    # 반환: (merged (S/m², Dout), deepstack [(S/m², Dout)] × len(indexes))
    # ------------------------------------------------------------------ #
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        indices, weights = bilinear_pos_embed_index(
            grid_thw, self.num_grid_per_side, self.spatial_merge_size
        )
        position_ids = vision_position_ids(grid_thw, self.spatial_merge_size)
        seq_lengths  = vision_seq_lengths(grid_thw)

        hidden_states = self.patch_embed(hidden_states)                     # (S, D)
        pos_embeds    = (self.pos_embed(indices) * weights[:, :, None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary = self.rotary_pos_emb(position_ids)                          # (S, hd/2)
        emb    = torch.cat((rotary, rotary), dim=-1)                        # (S, hd)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_features: List[torch.Tensor] = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, seq_lengths, position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)]
                deepstack_features.append(merger(hidden_states))

        return self.merger(hidden_states), deepstack_features
