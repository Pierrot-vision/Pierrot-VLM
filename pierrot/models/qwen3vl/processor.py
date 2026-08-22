"""Qwen3-VL 프로세서 (동적 해상도 패치화 + ChatML 프롬프트 토크나이즈), 추론 전용.

Qwen3-VL 은 이미지를 정사각 타일로 자르지 않는다. 원본 종횡비를 유지한 채
`smart_resize` 로 **32(=patch_size×spatial_merge_size) 배수**가 되도록 리사이즈하고,
전체 픽셀 수를 [min_pixels, max_pixels] 범위로 맞춘다. 그 결과 격자
(grid_t, grid_h, grid_w) 가 이미지마다 달라지고, 이미지 토큰 수도 함께 달라진다:
    이미지 토큰 수 = grid_t · grid_h · grid_w / spatial_merge_size²
따라서 **max_pixels 가 곧 시퀀스 길이·VRAM 예산**이다(가장 중요한 조절 손잡이).

한 샘플의 (input_ids, attention_mask, pixel_values, image_grid_thw)를 만든다:
  - 시퀀스(ChatML): <|im_start|>user\\n<|vision_start|><|image_pad|>×N<|vision_end|>{prefix}
                    <|im_end|>\\n<|im_start|>assistant\\n
  - assistant 턴 시작까지만 만들고 그 뒤를 모델이 생성한다. 정답을 붙여 labels 를
    만드는 학습 경로는 이 추론 배포본에 없다.

pixel_values 는 (다른 모델처럼 배치 축으로 패딩하지 않고) 배치 전체의 패치를 이어붙인
**패킹 텐서** (총패치수, patch_dim) 이다. 이미지 크기가 제각각이라 패딩이 불가능하고,
패딩이 없어 낭비도 없다. 이미지 경계는 image_grid_thw 가 알려 준다.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .config import Qwen3VLConfig

IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD  = [0.5, 0.5, 0.5]

IMAGE_TOKEN        = "<|image_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN   = "<|vision_end|>"
IM_START           = "<|im_start|>"
IM_END             = "<|im_end|>"

# 공식 preprocessor_config.json 기본값(= 256×256 ~ 4096×4096 픽셀).
# max_pixels 를 그대로 쓰면 이미지 한 장이 16384 토큰까지 나올 수 있어 학습에는 과하다.
OFFICIAL_MIN_PIXELS = 256 * 256
OFFICIAL_MAX_PIXELS = 4096 * 4096


# ------------------------------------------------------------------ #
# Qwen 계열 동적 해상도 리사이즈 규칙.
#   ① 세로/가로를 각각 factor 배수로 반올림
#   ② 총 픽셀이 max_pixels 를 넘으면 비율을 유지한 채 축소(내림)
#      min_pixels 보다 작으면 확대(올림)
# 종횡비가 200배를 넘으면(극단적 띠 이미지) 패치 격자가 무너지므로 오류를 낸다.
# ------------------------------------------------------------------ #
def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    if min(height, width) <= 0:
        raise ValueError(f"이미지 크기가 잘못되었습니다: {height}x{width}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"종횡비가 200 을 넘습니다: {max(height, width) / min(height, width):.1f}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta  = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta  = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return max(factor, h_bar), max(factor, w_bar)


class Qwen3VLProcessor:
    """동적 해상도 패치화 + ChatML 프롬프트/정답 토크나이즈 + 손실 라벨 생성."""

    # ------------------------------------------------------------------ #
    # 토크나이저와 config 를 받아 패치 크기·픽셀 예산·특수 토큰 id 를 캐시한다.
    #
    # min_pixels / max_pixels 를 None 으로 두면 공식 기본값을 쓴다. ★ max_pixels 는
    # 이미지 토큰 수(=시퀀스 길이)를 직접 결정하므로 학습·추론에서 반드시 같아야 하며,
    # 학습 산출물에는 sidecar(qwen3vl_preprocessor.json)로 동봉된다.
    #   대략적인 감: 이미지 토큰 수 ≈ max_pixels / (patch×merge)² = max_pixels / 1024
    #   (예: 1,310,720 → 최대 1280 패치 → 320 이미지 토큰)
    # system_prompt 를 주면 ChatML 맨 앞에 system 턴을 붙인다(공식 템플릿은 기본 없음).
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        tokenizer,
        config: Qwen3VLConfig,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ):
        vision = config.vision_config
        self.tokenizer           = tokenizer
        self.patch_size          = vision.patch_size
        self.temporal_patch_size = vision.temporal_patch_size
        self.merge_size          = vision.spatial_merge_size
        self.factor              = self.patch_size * self.merge_size
        self.min_pixels          = OFFICIAL_MIN_PIXELS if min_pixels is None else int(min_pixels)
        self.max_pixels          = OFFICIAL_MAX_PIXELS if max_pixels is None else int(max_pixels)
        self.system_prompt       = system_prompt

        if self.min_pixels <= 0 or self.max_pixels <= 0:
            raise ValueError("min_pixels / max_pixels 는 1 이상이어야 합니다.")
        if self.min_pixels > self.max_pixels:
            raise ValueError(f"min_pixels({self.min_pixels}) > max_pixels({self.max_pixels}) 입니다.")

        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.pad_token_id   = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                               else tokenizer.eos_token_id)
        # Qwen 은 <|im_end|> 로 턴을 끝내지만 <|endoftext|> 도 종료로 취급한다.
        im_end            = tokenizer.convert_tokens_to_ids(IM_END)
        self.eos_token_id = [i for i in {im_end, tokenizer.eos_token_id} if i is not None]

    # ------------------------------------------------------------------ #
    # PIL 이미지 → (패킹 패치 텐서 (n_patches, patch_dim), (t, h, w) 격자).
    #   ① smart_resize 로 32 배수 크기 결정 → BICUBIC 리사이즈
    #   ② /255 후 (x-0.5)/0.5 로 [-1,1] 정규화
    #   ③ 머저 블록(m×m) 우선 순서로 패치를 펼친다 — 이 순서 덕분에 비전 머저가
    #      reshape 한 번으로 이웃 패치를 합칠 수 있다(vision.py 참조).
    # 정지 이미지는 t=1 이며, Conv3d 의 시간 축(temporal_patch_size)은 같은 프레임을
    # 복제해 채운다(공식 전처리와 동일).
    # ------------------------------------------------------------------ #
    def process_image(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        image  = image.convert("RGB")
        w0, h0 = image.size
        h, w   = smart_resize(h0, w0, self.factor, self.min_pixels, self.max_pixels)
        image  = image.resize((w, h), resample=Image.Resampling.BICUBIC)

        arr = np.array(image, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)
        x   = torch.from_numpy(arr.transpose(2, 0, 1))                 # (C, H, W)

        p, m, tp = self.patch_size, self.merge_size, self.temporal_patch_size
        grid_h, grid_w = h // p, w // p
        x = x.reshape(3, grid_h // m, m, p, grid_w // m, m, p)
        x = x.permute(1, 4, 2, 5, 0, 3, 6)                             # (gh/m, gw/m, m, m, C, p, p)
        x = x.unsqueeze(5).expand(-1, -1, -1, -1, -1, tp, -1, -1)      # 시간 축 복제
        patches = x.reshape(grid_h * grid_w, 3 * tp * p * p).contiguous()
        return patches, (1, grid_h, grid_w)

    # ------------------------------------------------------------------ #
    # 전처리 설정을 산출물 디렉토리에 저장한다(qwen3vl_preprocessor.json).
    # 학습 때 쓴 픽셀 예산을 체크포인트에 동봉해, 다른 환경에서 로드해도 동일한
    # 전처리(=동일 이미지 토큰 수)를 복원할 수 있게 한다.
    # 학습 엔진이 이 훅으로 써 둔 파일을 weights.build_processor 가 되읽는다.
    # ------------------------------------------------------------------ #
    def save_preprocessor_config(self, path: str) -> None:
        import json
        import os

        cfg = {
            "min_pixels":    self.min_pixels,
            "max_pixels":    self.max_pixels,
            "patch_size":    self.patch_size,
            "merge_size":    self.merge_size,
            "system_prompt": self.system_prompt,
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "qwen3vl_preprocessor.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # 격자로부터 이미지 placeholder 문자열을 만든다:
    #   <|vision_start|> + <|image_pad|>×(t·h·w/m²) + <|vision_end|>
    # ------------------------------------------------------------------ #
    def _image_prompt_string(self, grid: Tuple[int, int, int]) -> str:
        n_tokens = grid[0] * grid[1] * grid[2] // (self.merge_size ** 2)
        return VISION_START_TOKEN + IMAGE_TOKEN * n_tokens + VISION_END_TOKEN

    # ------------------------------------------------------------------ #
    # 특수 토큰을 자동 추가하지 않고 텍스트를 토크나이즈한다
    # (ChatML 마커를 문자열로 직접 넣으므로 중복 추가를 막는다).
    # ------------------------------------------------------------------ #
    def _encode_text(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------------ #
    # 한 샘플을 인코딩한다.
    #   반환: {input_ids, attention_mask, pixel_values, image_grid_thw}
    #   assistant 턴 시작까지만 만든다(그 뒤를 모델이 생성한다).
    # image 가 None 이면 텍스트 전용 시퀀스가 된다(동등성 점검·순수 텍스트 replay 용).
    # ------------------------------------------------------------------ #
    def encode_one(self, image: Optional[Image.Image],
                   prefix: str) -> Dict[str, object]:
        patches, grid = (None, None) if image is None else self.process_image(image)
        image_string  = "" if image is None else self._image_prompt_string(grid)

        prompt = ""
        if self.system_prompt:
            prompt += f"{IM_START}system\n{self.system_prompt}{IM_END}\n"
        prompt += f"{IM_START}user\n{image_string}{prefix or ''}{IM_END}\n{IM_START}assistant\n"
        prompt_ids = self._encode_text(prompt)

        return {
            "pixel_values":   patches,
            "image_grid_thw": grid,
            "input_ids":      prompt_ids,
            "attention_mask": [1] * len(prompt_ids),
        }

    # ------------------------------------------------------------------ #
    # 추론 편의: 이미지·프롬프트 배치를 모델 입력 텐서 딕셔너리로 만든다(우측 패딩).
    # infer_qwen3vl.py 에서 사용.
    # ------------------------------------------------------------------ #
    def __call__(self, images: Sequence[Image.Image], text: Sequence[str]) -> Dict[str, torch.Tensor]:
        if len(images) != len(text):
            raise ValueError(f"이미지 수({len(images)})와 프롬프트 수({len(text)})가 같아야 합니다.")
        enc = [self.encode_one(img, t) for img, t in zip(images, text)]
        return collate_encoded(enc, self.pad_token_id, max_length=None)


# ------------------------------------------------------------------ #
# 인코딩된 샘플 리스트를 배치 텐서로 만든다.
#   input_ids/attention_mask          : (B, Lmax) 우측 패딩
#   pixel_values                      : (총패치수, patch_dim) 패킹(패딩 없음)
#   image_grid_thw                    : (이미지수, 3)
# 이미지가 하나도 없으면 pixel_values/image_grid_thw 를 넣지 않는다(텍스트 전용).
# max_length 초과 시 절단 없이 ValueError — 이미지 토큰이 잘리면 입력이 통째로 어긋난다.
# ------------------------------------------------------------------ #
def collate_encoded(enc: List[Dict[str, object]], pad_id: int,
                    max_length: Optional[int]) -> Dict[str, torch.Tensor]:
    lens = [len(e["input_ids"]) for e in enc]
    for n in lens:
        if max_length is not None and n > max_length:
            raise ValueError(
                f"시퀀스 길이({n}) > max_length({max_length}). max_pixels 를 낮춰 이미지 토큰 수를 "
                f"줄이거나 프롬프트 길이를 줄이세요(절단 시 입력 손상)."
            )
    B, L = len(enc), max(lens)

    input_ids      = torch.full((B, L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)

    for i, e in enumerate(enc):
        n = lens[i]
        input_ids[i, :n]      = torch.tensor(e["input_ids"], dtype=torch.long)
        attention_mask[i, :n] = 1

    out: Dict[str, torch.Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}

    # 이미지는 크기가 제각각이라 패딩하지 않고 배치 순서대로 이어붙인다.
    # (모델의 masked_scatter 가 같은 순서로 소비한다.)
    patches = [e["pixel_values"] for e in enc if e.get("pixel_values") is not None]
    if patches:
        out["pixel_values"]   = torch.cat(patches, dim=0)
        out["image_grid_thw"] = torch.tensor(
            [e["image_grid_thw"] for e in enc if e.get("image_grid_thw") is not None], dtype=torch.long
        )
    return out
