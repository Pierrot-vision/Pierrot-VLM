"""SmolVLM 멀티모달 커넥터 (픽셀셔플 + 선형 투영, 스크래치 구현).

비전 패치 시퀀스를 s²(scale_factor²) 배로 공간압축한 뒤, 늘어난 채널 차원을
언어 모델 hidden 차원으로 선형 투영한다. 압축으로 타일당 이미지 토큰 수가
(image_size/patch)² → (image_size/patch)²/s² 로 줄어 시퀀스가 짧아진다.

모듈 이름은 HF SmolVLM 체크포인트 키와 일치한다:
    model.connector.modality_projection.proj.weight
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import SmolVLM2Config


class SmolVLMSimpleMLP(nn.Module):
    """픽셀셔플로 늘어난 (vision_hidden × s²) 차원을 언어 hidden 으로 투영(bias 없음)."""

    def __init__(self, config: SmolVLM2Config):
        super().__init__()
        input_size  = config.vision_config.hidden_size * (config.scale_factor ** 2)
        output_size = config.text_config.hidden_size
        self.proj   = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SmolVLMConnector(nn.Module):
    """픽셀셔플 → 선형 투영 커넥터."""

    def __init__(self, config: SmolVLM2Config):
        super().__init__()
        self.scale_factor        = config.scale_factor
        self.modality_projection = SmolVLMSimpleMLP(config)

    # ------------------------------------------------------------------ #
    # 픽셀셔플: (B, N, D) 패치 시퀀스를 s² 배 압축해 (B, N/s², D·s²) 로 만든다.
    # N=한 변 h(=w) 패치의 정사각 격자라고 보고, s×s 이웃 패치를 채널로 접는다.
    # (HF Idefics3.pixel_shuffle 과 동일한 permute/reshape 순서)
    # ------------------------------------------------------------------ #
    def pixel_shuffle(self, x: torch.Tensor, scale_factor: int) -> torch.Tensor:
        bsz, seq, embed_dim = x.size()
        height = width = int(seq ** 0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor),
                      embed_dim * (scale_factor ** 2))
        x = x.permute(0, 2, 1, 3)
        return x.reshape(bsz, int(seq / (scale_factor ** 2)), embed_dim * (scale_factor ** 2))

    # ------------------------------------------------------------------ #
    # (B, N, vision_hidden) → 픽셀셔플 → (B, N/s², vision_hidden·s²) → 투영 →
    # (B, N/s², text_hidden). 반환 토큰 수 N/s² = config.image_seq_len.
    # ------------------------------------------------------------------ #
    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        return self.modality_projection(image_hidden_states)
