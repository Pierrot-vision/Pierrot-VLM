"""멀티모달 프로젝터: SigLIP 패치 임베딩 -> 언어 모델 hidden 차원 선형 사상."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import PaliGemma2Config


class MultiModalProjector(nn.Module):
    """비전 특징을 언어 모델 임베딩 차원으로 옮기는 단일 선형층."""

    # ------------------------------------------------------------------ #
    # vision_hidden -> projection_dim(=언어 hidden) 선형층을 만든다(bias 있음).
    # ------------------------------------------------------------------ #
    def __init__(self, config: PaliGemma2Config):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    # ------------------------------------------------------------------ #
    # (B, num_patches, vision_hidden) -> (B, num_patches, projection_dim).
    # 투영 결과가 언어 모델 임베딩과 같은 차원이 되어 시퀀스에 섞일 수 있다.
    # ------------------------------------------------------------------ #
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features)
