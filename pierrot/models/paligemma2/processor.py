"""PaliGemma2 프로세서 (스크래치, 추론 전용).

이미지 전처리 + `<image>` placeholder 를 포함한 프롬프트 토크나이즈를 담당한다.
  - prefix(프롬프트)까지만 만들고 그 뒤를 모델이 생성한다. 정답(suffix)을 붙여
    labels 를 만드는 학습 경로는 이 배포본에 없다.
  - 우측 패딩(right padding) 배치 처리

토크나이저는 HuggingFace 토크나이저(다운로드한 공개 모델의 것)를 그대로 사용하되,
<image> / <locNNNN> / <segNNN> 특수 토큰이 없으면 추가한다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
IMAGE_TOKEN = "<image>"


# ------------------------------------------------------------------ #
# 한 장의 PIL 이미지를 모델 입력 텐서 형태로 전처리한다.
# RGB 변환 → (size,size) BICUBIC 리사이즈 → /255 → (x-0.5)/0.5 정규화 →
# HWC→CHW 전치. 반환은 numpy (C,H,W).
# ------------------------------------------------------------------ #
def _process_image(image: Image.Image, size: int) -> np.ndarray:
    image = image.convert("RGB").resize((size, size), resample=Image.BICUBIC)
    arr   = np.array(image, dtype=np.float32) / 255.0
    arr   = (arr - np.array(IMAGENET_STANDARD_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STANDARD_STD, dtype=np.float32
    )
    return arr.transpose(2, 0, 1)  # HWC -> CHW


class PaliGemma2Processor:
    """이미지 전처리 + <image> placeholder 프롬프트 토크나이즈 + 라벨 생성."""

    # ------------------------------------------------------------------ #
    # 토크나이저를 받아 특수 토큰(<image>, <loc0000..1023>, <seg000..127>)을
    # 없으면 추가하고, <image>/pad/bos/eos 토큰 id 를 캐시한다.
    # ------------------------------------------------------------------ #
    def __init__(self, tokenizer, image_size: int = 224, num_image_tokens: int = 256):
        self.image_size       = image_size
        self.num_image_tokens = num_image_tokens

        # 특수 토큰 보강 (이미 있으면 no-op).
        extra = [IMAGE_TOKEN]
        extra += [f"<loc{i:04d}>" for i in range(1024)]   # detection 좌표 토큰
        extra += [f"<seg{i:03d}>" for i in range(128)]    # segmentation 토큰
        to_add = [t for t in extra if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
        if to_add:
            tokenizer.add_tokens(to_add)
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]}) \
            if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id else None

        self.tokenizer      = tokenizer
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.pad_token_id   = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_token_id   = tokenizer.bos_token_id
        self.eos_token_id   = tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    # 한 샘플의 (input_ids, token_type_ids) 를 만든다 (패딩 전).
    # 시퀀스: [<image>×N] + <bos> + prefix + "\n"  — 그 뒤를 모델이 생성한다.
    #   - prefix/이미지: token_type=0(양방향)
    # ------------------------------------------------------------------ #
    def _encode_one(self, prefix: str):
        image_ids = [self.image_token_id] * self.num_image_tokens
        # 프리픽스: <image>*N + <bos> + 프롬프트 + "\n"
        prefix_text_ids = self.tokenizer(prefix + "\n", add_special_tokens=False)["input_ids"]
        prefix_ids      = image_ids + [self.bos_token_id] + prefix_text_ids

        return list(prefix_ids), [0] * len(prefix_ids)     # 0 = prefix (양방향)

    # ------------------------------------------------------------------ #
    # 이미지·프롬프트 배치를 모델 입력 텐서 딕셔너리로 만든다.
    # 이미지를 stack 하고, 각 샘플을 _encode_one 으로 인코딩한 뒤 우측 패딩으로
    # 길이를 맞춘다. 반환 키:
    #   pixel_values / input_ids / attention_mask / token_type_ids
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        images: Sequence[Image.Image],
        text: Sequence[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assert len(images) == len(text), "이미지 수와 프롬프트 수가 같아야 합니다."

        # 이미지
        pixel_values = np.stack([_process_image(img, self.image_size) for img in images], axis=0)
        pixel_values = torch.tensor(pixel_values, dtype=torch.float32)

        # 텍스트
        enc = [self._encode_one(t) for t in text]
        # 길이 결정: 절단 없음. max_length 초과 시 조용히 자르지 않고 명시적 오류를 낸다.
        # (이미지 토큰이 잘리면 입력이 통째로 어긋나므로 조용한 절단은 금물이다.)
        seqs_len = []
        for ids, _ in enc:
            if max_length is not None and len(ids) > max_length:
                raise ValueError(
                    f"시퀀스 길이({len(ids)}) > max_length({max_length}). "
                    f"이미지 토큰 {self.num_image_tokens}개는 자를 수 없습니다 — "
                    f"max_length 를 {len(ids)} 이상으로 키우거나 프롬프트를 줄이세요."
                )
            seqs_len.append(len(ids))
        max_len = max(seqs_len)

        # 우측 패딩된 (B, max_len) 텐서들. pad=pad_token_id, mask=0 으로 채워 시작
        B              = len(enc)
        input_ids      = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        token_type_ids = torch.zeros((B, max_len), dtype=torch.long)

        for i, (ids, ttids) in enumerate(enc):
            n                     = seqs_len[i]
            input_ids[i, :n]      = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1
            token_type_ids[i, :n] = torch.tensor(ttids, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
