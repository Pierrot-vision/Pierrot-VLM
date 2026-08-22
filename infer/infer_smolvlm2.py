#!/usr/bin/env python
"""SmolVLM2 추론 엔트리포인트 (다른 모델 추론과 독립).

공개 사전학습 모델로 바로 추론:
    python infer/infer_smolvlm2.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
        --image path/to/img.jpg --prompt "Describe this image."

파인튜닝(검출) 결과로 추론 + 박스 시각화:
    python infer/infer_smolvlm2.py --model ./outputs/smolvlm2_ft/final --image img.jpg \
        --prompt "detect earrings ; necklaces ; ..." --detect --save-viz out.jpg
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
from pierrot.models.smolvlm2.detection import draw_detections, parse_detections  # noqa: E402
from pierrot.models.smolvlm2.weights import load_pretrained                   # noqa: E402

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_PARAMS = DEFAULTS["smolvlm2"]

# 기본 모델은 infer/defaults.py 의 pretrained (없으면 학습 산출물 final 디렉토리).
_DEFAULT_MODEL = default_model("smolvlm2")


# ------------------------------------------------------------------ #
# 추론 엔트리포인트.
# 모델/프로세서 로드 → 이미지·프롬프트 인코딩 → generate → 새 토큰만 디코드·출력.
# --model 기본값은 infer/defaults.py 의 pretrained(또는 학습 산출물 final).
# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLM2 추론")
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help="HF Hub id 또는 로컬 경로(파인튜닝 결과 포함). 기본값은 infer/defaults.py 의 pretrained")
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--prompt", default="Describe this image.", help="프롬프트(프리픽스)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", default=_PARAMS["dtype"], choices=list(_DTYPES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detect", action="store_true", help="출력을 검출(텍스트 좌표)로 파싱")
    parser.add_argument("--save-viz", default=None, help="검출 박스를 그려 저장할 경로")
    # 전처리(타일) 옵션 — 미지정(None)이면 모델의 sidecar(파인튜닝 산출물) → 공식값 순으로 상속.
    parser.add_argument("--max-splits-per-side", type=int, default=None,
                        help="한 변 최대 타일 수. 미지정 시 모델 sidecar/공식값 상속")
    parser.add_argument("--size-longest-edge", type=int, default=None,
                        help="분할 전 긴 변 크기. 미지정 시 모델 sidecar/공식값 상속")
    parser.add_argument("--image-splitting", dest="do_image_splitting", action="store_true", default=None,
                        help="타일 분할 강제 ON")
    parser.add_argument("--no-image-splitting", dest="do_image_splitting", action="store_false",
                        help="타일 분할 강제 OFF(글로벌 1장)")
    args = parser.parse_args()

    dtype = _DTYPES[args.dtype]
    # ★ 전처리 설정 우선순위(다운로드 후 build_processor 안에서 판정 — 로컬/Hub 동일):
    #   ① CLI 명시 → ② 파인튜닝 모델의 sidecar(자체완결) → ③ 공식 base 모델(sidecar 없음)이면
    #      infer/defaults.py 안전 기본값(fallback) → ④ 공식값 자동.
    #   CLI 미지정은 None 으로 넘겨 sidecar 에 위임하고, 안전 기본값은 fallback 으로 분리해
    #   Hub repo id 로 올린 파인튜닝 모델도 sidecar 가 안전 기본값을 덮지 않게 한다.
    model, processor = load_pretrained(
        args.model, device=args.device, dtype=dtype,
        do_image_splitting=args.do_image_splitting,     # None=미지정(sidecar 위임)
        max_splits_per_side=args.max_splits_per_side,   # None=미지정
        size_longest_edge=args.size_longest_edge,       # None=미지정
        fallback_proc_kwargs={                           # sidecar 없는 base 모델에만 적용
            "do_image_splitting":  _PARAMS.get("do_image_splitting", True),
            "max_splits_per_side": _PARAMS.get("max_splits_per_side"),
            "size_longest_edge":   _PARAMS.get("size_longest_edge"),
        },
    )
    model.eval()

    image  = Image.open(args.image).convert("RGB")
    inputs = processor(images=[image], text=[args.prompt])
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    generated = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"].to(dtype),
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=processor.eos_token_id,
    )

    # generate 는 전체 시퀀스를 반환한다 — 프롬프트 뒤 새 토큰만 디코드.
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[0, prompt_len:]
    text       = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(text)

    # 검출 모드: 텍스트 좌표를 원본 픽셀 박스로 파싱 → 출력(+ 시각화 저장).
    if args.detect or args.save_viz:
        w, h       = image.size
        detections = parse_detections(text, w, h)
        print(f"\n[detections] {len(detections)}개")
        for d in detections:
            x0, y0, x1, y1 = (round(v, 1) for v in d["box"])
            print(f"  {d['label']:20s} [{x0}, {y0}, {x1}, {y1}]")
        if args.save_viz:
            draw_detections(image, detections, args.save_viz)
            print(f"[viz] 저장: {args.save_viz}")


if __name__ == "__main__":
    main()
