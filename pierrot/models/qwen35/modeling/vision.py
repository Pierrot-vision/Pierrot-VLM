"""Qwen3.5 비전 인코더 (동적 해상도 ViT, 스크래치 — DeepStack 없음).

Qwen3.5 의 비전 타워는 Qwen3-VL 과 **블록 단위까지 동일한 구조**다(패치 임베딩 →
48×48 학습형 위치 임베딩 bilinear 보간 → 2D RoPE → 양방향 블록 스택 → 패치 머저).
유일한 차이는 **DeepStack 이 없다**는 것 — 중간층 특징을 뽑는 별도 머저가 없고,
비전 타워는 최종 머저 출력 하나만 낸다(config.json 의 deepstack_visual_indexes=[]).

그래서 검증된 Pierrot Qwen3-VL 스크래치 빌딩 블록(패치 임베딩·RoPE·블록·머저)을
그대로 재사용하고, 이 파일은 DeepStack 을 뺀 타워 조립만 담당한다
(프로세서·데이터셋을 qwen3vl 에서 재사용하는 것과 같은 원칙).

모듈/파라미터 이름은 HF Qwen3.5 체크포인트 키(model.visual.*)와 일치한다:
    model.visual.patch_embed.proj                 (Conv3d)
    model.visual.pos_embed                        (nn.Embedding, 48×48 격자)
    model.visual.blocks.N.{norm1,norm2}
    model.visual.blocks.N.attn.{qkv,proj}
    model.visual.blocks.N.mlp.{linear_fc1,linear_fc2}
    model.visual.merger.{norm,linear_fc1,linear_fc2}
    (deepstack_merger_list 없음)

텐서 차원 표기:
    S    = 총 패치 수, D = vision hidden_size, m = spatial_merge_size
    S/m² = 이미지 토큰 수, Dout = out_hidden_size(= 언어 hidden)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...qwen3vl.modeling.vision import (
    Qwen3VLVisionBlock,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
    bilinear_pos_embed_index,
    vision_position_ids,
    vision_seq_lengths,
)
from ..config import Qwen35VisionConfig


class Qwen35VisionModel(nn.Module):
    """Qwen3.5 비전 타워 본체 (HF 키 model.visual.* 에 대응, DeepStack 없음)."""

    # ------------------------------------------------------------------ #
    # 패치 임베딩 · 학습형 위치 임베딩 · 2D RoPE · 블록 스택 · 머저를 구성한다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        self.config             = config
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side  = config.num_grid_per_side

        self.patch_embed    = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed      = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)

    # ------------------------------------------------------------------ #
    # 패킹된 패치 시퀀스를 인코딩한다.
    #   ① 패치 임베딩 → ② 48×48 위치 임베딩을 격자에 bilinear 보간해 덧셈
    #   → ③ 2D RoPE(cos/sin) 준비 → ④ 블록 스택(이미지 경계로 어텐션 분리)
    #   → ⑤ 머저로 m² 압축.  반환: (S/m², Dout) 하나 — DeepStack 특징이 없다.
    # ------------------------------------------------------------------ #
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
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

        for block in self.blocks:
            hidden_states = block(hidden_states, seq_lengths, position_embeddings)

        return self.merger(hidden_states)
