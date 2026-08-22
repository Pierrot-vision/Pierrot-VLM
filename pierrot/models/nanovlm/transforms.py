"""nanoVLM 이미지 전처리 변환 (동적 리사이즈 + 타일 분할).

큰 이미지를 patch_size(=vit_img_size) 격자로 나눠 여러 타일로 만들고, 여러 타일이
생기면 전체 축소본(global) 타일을 맨 앞에 붙인다. 각 타일은 ViT 입력 한 장이 된다.
토큰 문자열(get_image_string)이 이 타일 격자와 정확히 대응한다.

원본 nanoVLM data/custom_transforms.py 의 충실한 포팅.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, resize


class DynamicResize(torch.nn.Module):
    """긴 변 ≤ max_side_len 이고 양변이 patch_size 로 나눠떨어지게 리사이즈(비율 유지)."""

    # ------------------------------------------------------------------ #
    # patch_size(p), 긴 변 상한(m), 강제 확대 여부, 보간법을 설정한다.
    # ------------------------------------------------------------------ #
    def __init__(self, patch_size: int, max_side_len: int, resize_to_max_side_len: bool = False,
                 interpolation: InterpolationMode = InterpolationMode.BICUBIC) -> None:
        super().__init__()
        self.p = int(patch_size)
        self.m = int(max_side_len)
        self.interpolation = interpolation
        self.resize_to_max_side_len = resize_to_max_side_len

    # ------------------------------------------------------------------ #
    # (h,w) → patch_size 배수의 목표 (h,w). 긴 변을 상한/배수에 맞추고 비율 유지.
    # ------------------------------------------------------------------ #
    def _get_new_hw(self, h: int, w: int) -> Tuple[int, int]:
        long, short = (w, h) if w >= h else (h, w)
        target_long = self.m if self.resize_to_max_side_len else min(self.m, math.ceil(long / self.p) * self.p)
        scale        = target_long / long
        target_short = max(math.ceil(short * scale / self.p) * self.p, self.p)
        return (target_short, target_long) if w >= h else (target_long, target_short)

    # ------------------------------------------------------------------ #
    # PIL 또는 (C,H,W)/(B,C,H,W) 텐서를 받아 같은 타입으로 리사이즈해 반환.
    # ------------------------------------------------------------------ #
    def forward(self, img: Union[Image.Image, torch.Tensor]):
        if isinstance(img, Image.Image):
            w, h = img.size
            new_h, new_w = self._get_new_hw(h, w)
            return resize(img, [new_h, new_w], interpolation=self.interpolation)
        if not torch.is_tensor(img):
            raise TypeError(f"DynamicResize 는 PIL 또는 텐서만 받습니다: {type(img)}")

        batched = img.ndim == 4
        if img.ndim not in (3, 4):
            raise ValueError(f"텐서는 (C,H,W)/(B,C,H,W) 여야 합니다: {img.shape}")
        imgs = img if batched else img.unsqueeze(0)
        _, _, h, w = imgs.shape
        new_h, new_w = self._get_new_hw(h, w)
        out = resize(imgs, [new_h, new_w], interpolation=self.interpolation)
        return out if batched else out.squeeze(0)


class SplitImage(torch.nn.Module):
    """(B,C,H,W) 를 patch_size 정사각 타일로 분할 → (B·n_h·n_w, C, p, p), 격자(n_h,n_w)."""

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.p = patch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        b, c, h, w = x.shape
        if h % self.p or w % self.p:
            raise ValueError(f"이미지 크기 {(h, w)} 가 patch_size {self.p} 로 나눠떨어지지 않습니다.")
        n_h, n_w = h // self.p, w // self.p
        patches  = rearrange(x, "b c (nh ph) (nw pw) -> (b nh nw) c ph pw", ph=self.p, pw=self.p)
        return patches, (n_h, n_w)


class GlobalAndSplitImages(torch.nn.Module):
    """타일 분할 후, 타일이 여럿이면 전체 축소본(global)을 맨 앞에 붙인다."""

    def __init__(self, patch_size: int):
        super().__init__()
        self.p        = patch_size
        self.splitter = SplitImage(patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        patches, grid = self.splitter(x)
        if grid == (1, 1):
            return patches, grid                      # 타일 1개면 global 미부착
        global_patch = resize(x, [self.p, self.p])
        return torch.cat([global_patch, patches], dim=0), grid
