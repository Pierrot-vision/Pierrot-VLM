"""멀티모달 프로젝터 (픽셀 셔플 + 선형).

SigLIP 패치 임베딩을 언어 모델 hidden 차원으로 옮긴다. 그 전에 '픽셀 셔플'로
공간 이웃 s×s 패치를 채널로 접어 토큰 수를 s² 배 줄인다(예: 1024→64).
→ 언어 모델에 들어가는 이미지 토큰 수를 줄여 시퀀스 길이를 절약한다.

텐서 차원 표기:
    B    = 이미지(타일) 수, N = 패치 수(예: (512/16)²=1024), Dv = vit_hidden_dim(768)
    s    = mp_pixel_shuffle_factor(4), N' = N/s²(예: 1024/16=64) = 타일당 이미지 토큰 수
    Din  = Dv·s²(프로젝터 입력), Dlm = lm_hidden_dim(언어 임베딩 차원)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    """픽셀 셔플로 토큰을 압축한 뒤 언어 hidden 으로 사상하는 프로젝터."""

    # ------------------------------------------------------------------ #
    # 입력 차원 = vit_hidden × (셔플배수)², 출력 = 언어 hidden. bias 없는 선형 1개.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg):
        super().__init__()
        self.cfg          = cfg
        self.input_dim    = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor ** 2)
        self.output_dim   = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    # 선형 가중치 normal(0,0.02) 초기화.
    # ------------------------------------------------------------------ #
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ #
    # (B, seq, embd) → (B, seq/s², embd·s²). seq 는 완전제곱수여야 하며
    # 한 변(seq_root)이 셔플배수로 나눠떨어져야 한다. 공간 s×s 블록을 채널로 접는다.
    # 참조: huggingface/smollm vllama3 modeling_vllama3.py
    # ------------------------------------------------------------------ #
    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, Dv)   N=패치 수(완전제곱수), Dv=vit_hidden
        bsz, seq, embed_dim = x.size()
        seq_root            = int(seq ** 0.5)                        # 한 변 길이 √N (예: √1024=32)
        assert seq_root ** 2 == seq, "픽셀 셔플: 시퀀스 길이가 완전제곱수가 아닙니다."
        assert seq_root % self.scale_factor == 0, "픽셀 셔플: seq_root 가 배수로 나눠떨어지지 않습니다."

        height = width = seq_root                                    # 패치 격자 h=w=√N
        x      = x.view(bsz, height, width, embed_dim)               # (B, h, w, Dv) 1D 시퀀스를 2D 격자로
        h_out  = height // self.scale_factor                         # 출력 격자 높이 h/s
        w_out  = width // self.scale_factor                          # 출력 격자 너비  w/s

        # s×s 이웃 블록을 채널축으로 접는다: (B, h,w, Dv) → (B, h/s, w/s, Dv·s²)
        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)  # (B, h/s, s, w/s, s, Dv)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()                # (B, h/s, w/s, s, s, Dv) — s,s 를 뒤로 모음
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor ** 2)  # (B, N/s², Dv·s²)
        return x

    # ------------------------------------------------------------------ #
    # 픽셀 셔플로 토큰을 s² 배 압축한 뒤 선형으로 언어 hidden 에 사상한다.
    #   (B, N, Dv) → 셔플 (B, N', Din=Dv·s²) → 선형 (B, N', Dlm)
    #   N' = mp_image_token_length (타일당 이미지 토큰 수)
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pixel_shuffle(x))                     # (B, N', Dlm)
