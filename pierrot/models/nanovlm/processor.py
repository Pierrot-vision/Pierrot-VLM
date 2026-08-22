"""nanoVLM 프로세서 (이미지 전처리 + 프롬프트 토크나이즈), 추론 전용.

토크나이저는 공개 SmolLM2 토크나이저에 66개 특수 토큰(<|image|>, <|global_image|>,
<row_i_col_j>)을 추가하고 ChatML 템플릿을 붙인 것을 쓴다. 이미지 전처리는 동적 리사이즈
+ 타일 분할이며, 각 타일마다 mp_image_token_length(64)개의 <|image|> 자리를 만든다.

한 샘플의 (input_ids, attention_mask, images)를 만든다:
  - 시퀀스: <|im_start|>user\n {이미지문자열}{prefix}<|im_end|>\n<|im_start|>assistant\n
  - assistant 턴 시작까지만 만들고 그 뒤를 모델이 생성한다.

encode_one 은 suffix 를 받는 경로도 남겨 둔다 — 학습용이 아니라 **우도(likelihood) 채점**
용이다. MMStar 같은 객관식 벤치는 선택지 글자의 '첫 답토큰 id' 가 필요한데, 그 id 를
얻으려면 정답 문자열을 같은 규칙으로 인코딩해 봐야 한다(eval/eval_nanovlm_bench.py).
학습용 다중턴 패킹(encode_chat)은 이 배포본에 없다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image

from .config import VLMConfig
from .transforms import DynamicResize, GlobalAndSplitImages

_TOKENIZERS_CACHE: dict = {}


# ------------------------------------------------------------------ #
# 공개 토크나이저에 특수 토큰/챗 템플릿을 주입해 로드(모듈 캐시). pad=eos 로 설정.
# extra_special_tokens 값은 tokenizer.image_token / global_image_token / r{i}c{j}
# 속성과 tokenizer.image_token_id 로도 접근된다.
# ------------------------------------------------------------------ #
def get_tokenizer(name: str, extra_special_tokens: Optional[dict] = None, chat_template: Optional[str] = None):
    key = (name, tuple(sorted((extra_special_tokens or {}).items())), chat_template)
    if key not in _TOKENIZERS_CACHE:
        from transformers import AutoTokenizer

        kwargs = {"use_fast": True}
        if extra_special_tokens is not None:
            kwargs["extra_special_tokens"] = extra_special_tokens
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        tok = AutoTokenizer.from_pretrained(name, **kwargs)
        tok.pad_token = tok.eos_token
        _TOKENIZERS_CACHE[key] = tok
    return _TOKENIZERS_CACHE[key]


# ------------------------------------------------------------------ #
# 이미지 전처리 파이프라인: 동적 리사이즈 → ToTensor → (global +) 타일 분할.
# ------------------------------------------------------------------ #
def get_image_processor(max_img_size: int, splitted_image_size: int, resize_to_max_side_len: bool = False):
    return transforms.Compose([
        DynamicResize(splitted_image_size, max_img_size, resize_to_max_side_len),
        transforms.ToTensor(),
        GlobalAndSplitImages(splitted_image_size),
    ])


# ------------------------------------------------------------------ #
# 타일 격자 목록 → <|image|> 플레이스홀더 문자열. 타일마다 mp_image_token_length 개.
# global_image_token 이 있으면 전체 축소본 블록을 먼저 넣는다(타일 1개면 그것만).
# ------------------------------------------------------------------ #
def get_image_string(tokenizer, splitted_image_counts: List[Tuple[int, int]], mp_image_token_length: int) -> str:
    s = ""
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            s += f"<image: {idx}>"
        if hasattr(tokenizer, "global_image_token"):
            s += tokenizer.global_image_token + tokenizer.image_token * mp_image_token_length
            if n_h == 1 and n_w == 1:
                continue
        for i in range(n_h):
            for j in range(n_w):
                s += getattr(tokenizer, f"r{i + 1}c{j + 1}") + tokenizer.image_token * mp_image_token_length
    return s


class NanoVLMProcessor:
    """이미지 타일화 + ChatML 프롬프트/정답 토크나이즈 + 손실 라벨 생성."""

    # ------------------------------------------------------------------ #
    # cfg 로 토크나이저·이미지 프로세서를 만들고 특수 토큰 id 를 캐시한다.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: VLMConfig):
        self.cfg       = cfg
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
        self.image_processor = get_image_processor(cfg.max_img_size, cfg.vit_img_size, cfg.resize_to_max_side_len)
        self.mp_image_token_length = cfg.mp_image_token_length

        self.image_token_id = self.tokenizer.image_token_id
        self.eos_token_id   = self.tokenizer.eos_token_id
        self.pad_token_id   = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_token_id

    # ------------------------------------------------------------------ #
    # PIL 이미지 → (타일 텐서 (n_tiles,3,p,p), 격자 (n_h,n_w)).
    # 토크나이저에 global 토큰이 없는데 global 타일이 생겼으면 제거(안전장치).
    # ------------------------------------------------------------------ #
    def process_image(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        tiles, grid = self.image_processor(image.convert("RGB"))
        if not hasattr(self.tokenizer, "global_image_token") and grid[0] * grid[1] == len(tiles) - 1:
            tiles = tiles[1:]
        return tiles, grid

    # ------------------------------------------------------------------ #
    # ChatML 로 프롬프트를 만든다. add_generation_prompt=True 면 assistant 시작까지
    # (정답 제외) 반환 — 이 길이가 라벨 마스킹 경계(prefix_len)가 된다.
    # ------------------------------------------------------------------ #
    def _apply_template(self, user_content: str, assistant: Optional[str], add_generation_prompt: bool) -> List[int]:
        messages = [{"role": "user", "content": user_content}]
        if assistant is not None:
            messages.append({"role": "assistant", "content": assistant})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_special_tokens=False, add_generation_prompt=add_generation_prompt)

    # ------------------------------------------------------------------ #
    # 한 샘플을 인코딩한다.
    #   suffix 無(생성): {input_ids, attention_mask, images}  (assistant 시작 프롬프트까지)
    #   suffix 有(우도 채점): {input_ids, attention_mask, labels, images}
    #     - labels: prefix/이미지 = -100, assistant(정답) 토큰만 정답 id.
    #       첫 정답 토큰 id 를 찾는 용도이며 손실 계산에는 쓰지 않는다.
    # ------------------------------------------------------------------ #
    def encode_one(self, image: Image.Image, prefix: str, suffix: Optional[str] = None) -> Dict[str, object]:
        tiles, grid  = self.process_image(image)
        image_string = get_image_string(self.tokenizer, [grid], self.mp_image_token_length)
        user_content = image_string + (prefix or "")

        prompt_ids = self._apply_template(user_content, None, add_generation_prompt=True)

        if suffix is None:
            return {
                "input_ids": prompt_ids,
                "attention_mask": [1] * len(prompt_ids),
                "images": tiles,
            }

        full_ids   = self._apply_template(user_content, suffix, add_generation_prompt=False)
        prefix_len = len(prompt_ids)
        # 일반적으로 prompt_ids 는 full_ids 의 접두이나, 토크나이즈 불일치 대비 공통 접두 길이로 보정.
        if full_ids[:prefix_len] != prompt_ids:
            prefix_len = 0
            for a, b in zip(prompt_ids, full_ids):
                if a != b:
                    break
                prefix_len += 1

        labels = [-100] * prefix_len + list(full_ids[prefix_len:])
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "images": tiles,
        }
