#!/usr/bin/env python
"""nanoVLM 추론 엔트리포인트 (다른 모델 추론과 독립).

공개 사전학습 모델로 바로 추론:
    python infer/infer_nanovlm.py --model lusxvr/nanoVLM-450M \
        --image assets/image.png --prompt "What is this?"

학습 산출물로 추론:
    python infer/infer_nanovlm.py --model ./outputs/nanovlm_mix/final \
        --image img.jpg --prompt "질문"
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image

# 이 스크립트는 infer/ 하위에 있으므로, 최상위의 pierrot 패키지를 import 하려면
# repo 루트를 sys.path 에 넣어야 한다(직접 실행 시 script 디렉토리만 잡히므로).
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from infer.defaults import DEFAULTS, default_model                          # noqa: E402
from pierrot.models.nanovlm.weights import load_pretrained                  # noqa: E402

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_PARAMS = DEFAULTS["nanovlm"]

# 기본 모델은 infer/defaults.py 의 pretrained (없으면 학습 산출물 final 디렉토리).
_DEFAULT_MODEL = default_model("nanovlm")


# ------------------------------------------------------------------ #
# 추론 엔트리포인트.
# 모델/프로세서 로드 → 이미지·프롬프트 인코딩(ChatML) → generate → 새 토큰만 디코드.
# --model 기본값은 infer/defaults.py 의 pretrained(또는 학습 산출물 final).
# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(description="nanoVLM 추론")
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help="HF Hub id 또는 로컬 경로(파인튜닝 결과 포함). 기본값은 infer/defaults.py 의 pretrained")
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--prompt", default="What is this?", help="프롬프트")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--greedy", action="store_true", help="greedy 디코딩(기본은 top-k/p 샘플링)")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--dtype", default=_PARAMS["dtype"], choices=list(_DTYPES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype            = _DTYPES[args.dtype]
    model, processor = load_pretrained(args.model, device=args.device, dtype=dtype)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    enc   = processor.encode_one(image, args.prompt, suffix=None)   # 정답 없이 프롬프트만
    input_ids      = torch.tensor([enc["input_ids"]], device=args.device)
    attention_mask = torch.tensor([enc["attention_mask"]], device=args.device)
    images         = [enc["images"].to(args.device)]

    generated = model.generate(
        input_ids=input_ids,
        images=images,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        eos_token_id=processor.eos_token_id,
    )

    # generate 는 새로 생성된 토큰만 반환한다.
    text = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    print(text)


if __name__ == "__main__":
    main()
