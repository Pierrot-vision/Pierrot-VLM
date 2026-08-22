"""SmolVLM2 프로세서 (이미지 타일 분할 + 프롬프트 토크나이즈), 추론 전용.

SmolVLM(Idefics3)은 큰 이미지를 image_size 정사각 타일 격자로 나누고, 전체 축소본
(global) 한 장을 덧붙여 본다. 각 타일/글로벌은 커넥터를 거쳐 image_seq_len 개의
<image> placeholder 로 표현되며, 타일 앞에는 <fake_token_around_image> 와
<row_i_col_j>(또는 <global-img>) 마커가 붙는다.

한 샘플의 (input_ids, attention_mask, pixel_values)를 만든다:
  - 시퀀스: User:{이미지문자열}{prefix}<end_of_utterance>\nAssistant:
  - Assistant 턴 시작까지만 만들고 그 뒤를 모델이 생성한다.

encode_one 은 suffix 를 받는 경로도 남겨 둔다 — 학습용이 아니라 **우도(likelihood) 채점**
용이다. MMStar 같은 객관식 벤치는 선택지 글자의 '첫 답토큰 id' 가 필요한데, 그 id 를
얻으려면 정답 문자열을 같은 규칙으로 인코딩해 봐야 한다(eval/eval_smolvlm2_bench.py).
학습용 다중턴 패킹(encode_chat)은 이 배포본에 없다.

토크나이저는 공개 SmolVLM2 체크포인트의 것(특수 토큰·챗포맷 내장)을 그대로 쓴다.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .config import SmolVLM2Config

IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD  = [0.5, 0.5, 0.5]

IMAGE_TOKEN       = "<image>"
FAKE_IMAGE_TOKEN  = "<fake_token_around_image>"
GLOBAL_IMG_TOKEN  = "<global-img>"
END_OF_UTTERANCE  = "<end_of_utterance>"


# ------------------------------------------------------------------ #
# 한 장의 PIL 이미지를 (size,size) 정사각으로 리사이즈·정규화해 (3,size,size)
# 텐서로 만든다. /255 후 (x-0.5)/0.5 → [-1,1]. (정규화로 실제 픽셀은 0 이 되지
# 않아, 배치 정렬용 zero-패딩 타일과 구분된다.)
# ------------------------------------------------------------------ #
def _to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), resample=Image.Resampling.LANCZOS)
    arr   = np.array(image, dtype=np.float32) / 255.0
    arr   = (arr - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1))            # HWC -> CHW


# ------------------------------------------------------------------ #
# 분할 타일 격자용 이미지 프롬프트 문자열(HF Idefics3 get_image_prompt_string 동일).
#   행×열 타일: <fake><row_i_col_j><image>×L ... 줄바꿈 ...
#   마지막에 전체 축소본: <fake><global-img><image>×L<fake>
# ------------------------------------------------------------------ #
def _prompt_split_image(n_rows: int, n_cols: int, seq_len: int) -> str:
    s = ""
    for h in range(n_rows):
        for w in range(n_cols):
            s += FAKE_IMAGE_TOKEN + f"<row_{h + 1}_col_{w + 1}>" + IMAGE_TOKEN * seq_len
        s += "\n"
    s += "\n" + FAKE_IMAGE_TOKEN + GLOBAL_IMG_TOKEN + IMAGE_TOKEN * seq_len + FAKE_IMAGE_TOKEN
    return s


# ------------------------------------------------------------------ #
# 단일(분할 없음) 이미지 프롬프트 문자열: <fake><global-img><image>×L<fake>.
# ------------------------------------------------------------------ #
def _prompt_single_image(seq_len: int) -> str:
    return FAKE_IMAGE_TOKEN + GLOBAL_IMG_TOKEN + IMAGE_TOKEN * seq_len + FAKE_IMAGE_TOKEN


class SmolVLM2Processor:
    """이미지 타일화 + SmolVLM 프롬프트/정답 토크나이즈 + 손실 라벨 생성."""

    # ------------------------------------------------------------------ #
    # 토크나이저와 config 를 받아 타일 크기·이미지 토큰 수·특수 토큰 id 를 캐시한다.
    # do_image_splitting=False 면 항상 단일(글로벌) 타일만 쓴다(길이·메모리 절약).
    #
    # size_longest_edge / max_splits_per_side 를 None 으로 두면 공식 설정을 자동 도출한다:
    #   - size_longest_edge  = image_size × 4  (공식 SmolVLM 전 크기: 384→1536, 512→2048)
    #   - max_splits_per_side = size_longest_edge // image_size (=4, 공식 최대 타일 수)
    # 정수를 주면 그 값으로 상한을 둔다(메모리 절약용). build_processor 는 체크포인트의
    # preprocessor_config.json 이 있으면 그 size.longest_edge 를 우선 넘겨 공식값을 보장한다.
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        tokenizer,
        config: SmolVLM2Config,
        do_image_splitting: bool = True,
        max_splits_per_side: Optional[int] = None,
        size_longest_edge: Optional[int] = None,
    ):
        self.tokenizer           = tokenizer
        self.image_size          = config.vision_config.image_size
        self.image_seq_len       = config.image_seq_len
        self.do_image_splitting  = do_image_splitting
        # None → 공식 도출. 분할 전 이미지의 긴 변을 이 크기로 정규화(공식 size.longest_edge).
        self.size_longest_edge   = size_longest_edge if size_longest_edge is not None else self.image_size * 4
        self.max_splits_per_side = (max_splits_per_side if max_splits_per_side is not None
                                    else self.size_longest_edge // self.image_size)
        if self.max_splits_per_side <= 0:
            raise ValueError("max_splits_per_side 는 1 이상이어야 합니다.")
        if self.size_longest_edge <= 0:
            raise ValueError("size_longest_edge 는 1 이상이어야 합니다.")

        # 특수 토큰 보강(스크래치용 base 토크나이저 대비). 이미 있으면 no-op.
        specials = [IMAGE_TOKEN, FAKE_IMAGE_TOKEN, GLOBAL_IMG_TOKEN, END_OF_UTTERANCE]
        specials += [f"<row_{i}_col_{j}>" for i in range(1, 9) for j in range(1, 9)]
        missing = [t for t in specials if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
        if missing:
            tokenizer.add_special_tokens({"additional_special_tokens": missing})

        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.eou            = END_OF_UTTERANCE
        eou_id              = tokenizer.convert_tokens_to_ids(END_OF_UTTERANCE)
        self.eos_token_id   = eou_id if eou_id != tokenizer.unk_token_id else tokenizer.eos_token_id
        self.bos_token_id   = tokenizer.bos_token_id
        self.pad_token_id   = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else \
            (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)

    # ------------------------------------------------------------------ #
    # PIL 이미지 → (타일 텐서 리스트, (n_rows, n_cols)).
    #   ① 긴 변을 size_longest_edge 로 정규화(비율 유지) — 원본 픽셀이 아니라 이 기준으로
    #      타일 수를 정한다(공식 전처리 근접, 크기 불변성).
    #   ② n_rows/n_cols = ceil(정규화변/image_size), max_splits_per_side 로 상한.
    #   ③ 분할 없음/작은 이미지: [글로벌] 한 장, grid=(0,0)
    #      분할: 격자 크기로 리사이즈 후 image_size 정사각 타일 + 마지막에 글로벌, grid=(n_rows,n_cols)
    # 타일 순서는 프롬프트 문자열(_prompt_split_image)과 정확히 일치한다.
    #
    # 공식 전처리와 같이 격자에 맞춰 리사이즈하므로 각 타일은 image_size 정사각이고,
    # 비전 패치가 모두 유효해 타일 내부 pixel_attention_mask 는 필요하지 않다.
    # ------------------------------------------------------------------ #
    def process_image(self, image: Image.Image) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        image = image.convert("RGB")
        size  = self.image_size

        # ① 공식 SmolVLM 과 동일하게 긴 변을 목표 크기로 맞추고, 짧은 변은 짝수로
        #    올림한다. 두 단계 resize 를 유지해야 공개 체크포인트 processor 와 픽셀이 맞는다.
        w0, h0 = image.size
        aspect = w0 / h0
        if w0 >= h0:
            nw = self.size_longest_edge
            nh = max(1, int(nw / aspect))
            nh += nh % 2
        else:
            nh = self.size_longest_edge
            nw = max(1, int(nh * aspect))
            nw += nw % 2
        resized_longest = image.resize((nw, nh), resample=Image.Resampling.LANCZOS)

        # ② 비전 타일 크기의 배수로 올림한다. 공식 구현처럼 긴 축을 먼저 올린 뒤
        #    그 크기와 종횡비로 짧은 축을 계산한다.
        aspect = nw / nh
        if nw >= nh:
            grid_w = math.ceil(nw / size) * size
            grid_h = math.ceil(int(grid_w / aspect) / size) * size
        else:
            grid_h = math.ceil(nh / size) * size
            grid_w = math.ceil(int(grid_h * aspect) / size) * size

        n_cols = min(max(1, grid_w // size), self.max_splits_per_side)
        n_rows = min(max(1, grid_h // size), self.max_splits_per_side)
        if not self.do_image_splitting or n_rows * n_cols <= 1:
            return [_to_tensor(resized_longest, size)], (0, 0)

        # ③ 격자 크기에 맞춰 리사이즈 후 정확히 size×size 타일로 분할.
        resized = resized_longest.resize(
            (n_cols * size, n_rows * size), resample=Image.Resampling.LANCZOS
        )
        tiles: List[torch.Tensor] = []
        for r in range(n_rows):
            for c in range(n_cols):
                crop = resized.crop((c * size, r * size, (c + 1) * size, (r + 1) * size))
                tiles.append(_to_tensor(crop, size))
        tiles.append(_to_tensor(resized, size))               # 마지막에 글로벌 축소본
        return tiles, (n_rows, n_cols)

    # ------------------------------------------------------------------ #
    # 전처리 설정을 산출물 디렉토리에 저장한다(smolvlm2_preprocessor.json).
    # 학습 때 쓴 타일 설정을 체크포인트에 동봉해, 다른 환경에서 로드해도 동일한
    # 전처리를 복원할 수 있게 한다(build_processor 가 명시 인자 없을 때 이 파일을 읽음).
    # 학습 엔진이 이 훅으로 써 둔 파일을 weights.build_processor 가 되읽는다.
    # ------------------------------------------------------------------ #
    def save_preprocessor_config(self, path: str) -> None:
        import json
        import os

        cfg = {
            "do_image_splitting":  self.do_image_splitting,
            "size_longest_edge":   self.size_longest_edge,
            "max_splits_per_side": self.max_splits_per_side,
            "image_size":          self.image_size,
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "smolvlm2_preprocessor.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # 타일 격자로부터 <image> placeholder 를 포함한 이미지 프롬프트 문자열을 만든다.
    # ------------------------------------------------------------------ #
    def _image_prompt_string(self, n_rows: int, n_cols: int) -> str:
        if n_rows == 0 and n_cols == 0:
            return _prompt_single_image(self.image_seq_len)
        return _prompt_split_image(n_rows, n_cols, self.image_seq_len)

    # ------------------------------------------------------------------ #
    # 특수 토큰 없이 텍스트를 토크나이즈하고, bos=True 면 bos 토큰을 앞에 붙인다.
    # ------------------------------------------------------------------ #
    def _encode_text(self, text: str, bos: bool) -> List[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if bos and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return ids

    # ------------------------------------------------------------------ #
    # 한 샘플을 인코딩한다.
    #   suffix 無(생성): {input_ids, attention_mask, pixel_values}  (Assistant: 시작까지)
    #   suffix 有(우도 채점): {input_ids, attention_mask, labels, pixel_values}
    #     - labels: prefix/이미지 = -100, assistant(정답) 토큰만 정답 id.
    #       첫 정답 토큰 id 를 찾는 용도이며 손실 계산에는 쓰지 않는다.
    # pixel_values 는 타일 텐서 리스트 [(3,size,size), ...].
    # ------------------------------------------------------------------ #
    def encode_one(self, image: Image.Image, prefix: str, suffix: Optional[str] = None) -> Dict[str, object]:
        tiles, grid  = self.process_image(image)
        image_string = self._image_prompt_string(*grid)
        user_content = image_string + (prefix or "")

        prompt_text = f"User:{user_content}{self.eou}\nAssistant:"
        prompt_ids  = self._encode_text(prompt_text, bos=True)

        if suffix is None:
            return {
                "input_ids": prompt_ids,
                "attention_mask": [1] * len(prompt_ids),
                "pixel_values": tiles,
            }

        answer_ids = self._encode_text(f"{suffix}{self.eou}", bos=False)
        input_ids  = prompt_ids + answer_ids
        labels     = [-100] * len(prompt_ids) + list(answer_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "pixel_values": tiles,
        }

    # ------------------------------------------------------------------ #
    # 추론 편의: 이미지·프롬프트 배치를 (input_ids, attention_mask, pixel_values)
    # 텐서 딕셔너리로 만든다(우측 패딩). infer_smolvlm2.py 에서 사용.
    # ------------------------------------------------------------------ #
    def __call__(self, images: Sequence[Image.Image], text: Sequence[str]) -> Dict[str, torch.Tensor]:
        if len(images) != len(text):
            raise ValueError(f"이미지 수({len(images)})와 프롬프트 수({len(text)})가 같아야 합니다.")
        enc = [self.encode_one(img, t, suffix=None) for img, t in zip(images, text)]
        return collate_encoded(enc, self.pad_token_id, max_length=None)


# ------------------------------------------------------------------ #
# 인코딩된 샘플 리스트를 우측 패딩 배치 텐서로 만든다.
#   input_ids/attention_mask/(labels) : (B, Lmax) 우측 패딩
#   pixel_values                      : (B, max_tiles, 3, size, size) zero 패딩
# max_length 초과 시 절단 없이 ValueError(검출 라벨 손상 방지).
# ------------------------------------------------------------------ #
def collate_encoded(enc: List[Dict[str, object]], pad_id: int, max_length: Optional[int]) -> Dict[str, torch.Tensor]:
    has_labels = "labels" in enc[0]
    lens       = [len(e["input_ids"]) for e in enc]
    for n in lens:
        if max_length is not None and n > max_length:
            raise ValueError(
                f"시퀀스 길이({n}) > max_length({max_length}). 이미지 분할 수/프롬프트·정답 "
                f"길이를 줄이거나 max_length 를 키우세요(절단 시 라벨 손상)."
            )
    B, L = len(enc), max(lens)

    input_ids      = torch.full((B, L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    labels         = torch.full((B, L), -100, dtype=torch.long) if has_labels else None

    tile_counts = [len(e["pixel_values"]) for e in enc]
    max_tiles   = max(tile_counts)
    size        = enc[0]["pixel_values"][0].shape[-1]
    pixel_values = torch.zeros((B, max_tiles, 3, size, size), dtype=torch.float32)

    for i, e in enumerate(enc):
        n = lens[i]
        input_ids[i, :n]      = torch.tensor(e["input_ids"], dtype=torch.long)
        attention_mask[i, :n] = 1
        if has_labels:
            labels[i, :n] = torch.tensor(e["labels"], dtype=torch.long)
        for j, tile in enumerate(e["pixel_values"]):
            pixel_values[i, j] = tile

    out = {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}
    if has_labels:
        out["labels"] = labels
    return out
