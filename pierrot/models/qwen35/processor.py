"""Qwen3.5 이미지/ChatML 전처리.

Qwen3.5는 Qwen3-VL과 같은 patch/grid 입력 계약을 사용하므로 검증된 Pierrot
전처리(동적 해상도 패치화·placeholder·collate)를 재사용한다. 토크나이저와
vision config 값은 Qwen3.5 체크포인트에서 온다.

★ Qwen3-VL 과 다른 한 가지 — **thinking 채팅 템플릿**. Qwen3.5 공식 템플릿은
generation prompt(`<|im_start|>assistant\n`) 뒤에 thinking 모드에 따라 접두를 붙인다:
    enable_thinking=True  → "<think>\n"                (모델이 사고 과정을 이어 씀)
    enable_thinking=False → "<think>\n\n</think>\n\n"  (빈 사고 블록 — 바로 답변)
이 접두가 없으면 사전학습 프롬프트 분포와 어긋나 추론 품질이 떨어진다. 그래서
encode_one 을 오버라이드해 접두를 넣는다.

기본값은 enable_thinking=False(빈 사고 블록 뒤 바로 답변). 파인튜닝 산출물을 돌릴
때는 **학습에 쓴 모드와 같아야** 하며, 학습 산출물의 sidecar
(qwen3vl_preprocessor.json)에 그 값이 동봉되어 자동 상속된다.
"""

from __future__ import annotations

from typing import Dict, Optional

from PIL import Image

from ..qwen3vl.processor import (
    IM_END,
    IM_START,
    Qwen3VLProcessor,
    collate_encoded,
    smart_resize,
)

THINK_PREFIX_ON  = "<think>\n"              # thinking 모드: 모델이 사고 과정부터 생성
THINK_PREFIX_OFF = "<think>\n\n</think>\n\n"  # non-thinking: 빈 사고 블록 뒤 바로 답변


class Qwen35Processor(Qwen3VLProcessor):
    """Qwen3-VL 전처리 + Qwen3.5 thinking 템플릿 접두."""

    # ------------------------------------------------------------------ #
    # enable_thinking 만 추가된다(기본 False = 빈 사고 블록). None 은 False 로
    # 정규화한다 — weights.build_processor 의 "미지정 → sidecar 상속" 규약 때문.
    # ------------------------------------------------------------------ #
    def __init__(self, tokenizer, config, min_pixels=None, max_pixels=None,
                 system_prompt=None, enable_thinking: Optional[bool] = None):
        super().__init__(tokenizer, config, min_pixels=min_pixels,
                         max_pixels=max_pixels, system_prompt=system_prompt)
        self.enable_thinking = bool(enable_thinking)

    # ------------------------------------------------------------------ #
    # 공식 템플릿이 generation prompt 뒤에 붙이는 thinking 접두.
    # ------------------------------------------------------------------ #
    def _think_prefix(self) -> str:
        return THINK_PREFIX_ON if self.enable_thinking else THINK_PREFIX_OFF

    # ------------------------------------------------------------------ #
    # 한 샘플을 인코딩한다 — qwen3vl 과 같은 구조에 thinking 접두만 추가:
    #   <|im_start|>user\n{이미지}{prefix}<|im_end|>\n<|im_start|>assistant\n{think접두}
    #   접두까지가 프롬프트이고(공식 add_generation_prompt 와 동일) 그 뒤를 모델이 생성한다.
    # ------------------------------------------------------------------ #
    def encode_one(self, image: Optional[Image.Image],
                   prefix: str) -> Dict[str, object]:
        patches, grid = (None, None) if image is None else self.process_image(image)
        image_string  = "" if image is None else self._image_prompt_string(grid)

        prompt = ""
        if self.system_prompt:
            prompt += f"{IM_START}system\n{self.system_prompt}{IM_END}\n"
        prompt += (f"{IM_START}user\n{image_string}{prefix or ''}{IM_END}\n"
                   f"{IM_START}assistant\n{self._think_prefix()}")
        prompt_ids = self._encode_text(prompt)

        return {
            "pixel_values":   patches,
            "image_grid_thw": grid,
            "input_ids":      prompt_ids,
            "attention_mask": [1] * len(prompt_ids),
        }

    # ------------------------------------------------------------------ #
    # sidecar 에 enable_thinking 도 동봉한다 — 학습 때 쓴 템플릿 모드를 추론이
    # 자동 상속해야 프롬프트 분포가 일치한다(픽셀 예산과 같은 원리).
    # 이 배포본은 sidecar 를 읽기만 하지만, 재저장 경로도 대칭을 위해 남겨 둔다.
    # ------------------------------------------------------------------ #
    def save_preprocessor_config(self, path: str) -> None:
        import json
        import os

        super().save_preprocessor_config(path)
        sidecar = os.path.join(path, "qwen3vl_preprocessor.json")
        with open(sidecar, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["enable_thinking"] = self.enable_thinking
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)


__all__ = ["Qwen35Processor", "collate_encoded", "smart_resize",
           "THINK_PREFIX_ON", "THINK_PREFIX_OFF"]
