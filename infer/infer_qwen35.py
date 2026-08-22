#!/usr/bin/env python
"""Qwen3.5 이미지 추론 엔트리포인트 (다른 모델 추론과 독립).

공개 사전학습 모델로 바로 추론:
    python infer/infer_qwen35.py --model Qwen/Qwen3.5-4B \
        --image path/to/img.jpg --prompt "Describe this image."

파인튜닝 결과로 추론:
    python infer/infer_qwen35.py --model ./outputs/qwen35_fashion/final --image img.jpg
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer.defaults import DEFAULTS, default_model                          # noqa: E402
from pierrot.models.qwen35.weights import load_pretrained                     # noqa: E402

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_PARAMS = DEFAULTS["qwen35"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 이미지 추론")
    parser.add_argument("--model", default=default_model("qwen35"),
                        help="HF Hub id 또는 로컬 경로(파인튜닝 결과 포함). 기본값은 infer/defaults.py 의 pretrained")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=list(_DTYPES), default=_PARAMS["dtype"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-pixels", type=int, default=_PARAMS["max_pixels"])
    parser.add_argument("--min-pixels", type=int, default=_PARAMS["min_pixels"])
    args = parser.parse_args()

    dtype = _DTYPES[args.dtype]
    model, processor = load_pretrained(
        args.model, device=args.device, dtype=dtype,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        system_prompt=_PARAMS["system_prompt"],
        enable_thinking=_PARAMS["enable_thinking"],  # 학습과 같은 템플릿 모드(단일 소스)
    )
    model.eval()
    inputs = processor([Image.open(args.image).convert("RGB")], [args.prompt])
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=processor.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[-1]
    print(processor.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
