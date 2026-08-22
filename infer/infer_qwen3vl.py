#!/usr/bin/env python
"""Qwen3-VL 추론 엔트리포인트 (다른 모델 추론과 독립).

공개 사전학습 모델로 바로 추론:
    python infer/infer_qwen3vl.py --model Qwen/Qwen3-VL-2B-Instruct \
        --image path/to/img.jpg --prompt "Describe this image."

파인튜닝(검출) 결과로 추론 + 박스 시각화:
    python infer/infer_qwen3vl.py --model ./outputs/qwen3vl_ft/final --image img.jpg \
        --prompt "detect earrings ; necklaces ; ..." --detect --save-viz out.jpg

패션 속성(부위별 JSON) 파인튜닝 결과로 추론 — roi 를 픽셀 좌표로 복원해 출력:
    python infer/infer_qwen3vl.py --model ./outputs/qwen3vl_fashion/final \
        --image img.jpg --fashion
"""

from __future__ import annotations

import argparse
import json

import torch
from PIL import Image

# 이 스크립트는 infer/ 하위에 있으므로, 최상위의 pierrot 패키지를 import 하려면
# repo 루트를 sys.path 에 넣어야 한다(직접 실행 시 script 디렉토리만 잡히므로).
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from infer.defaults import DEFAULTS, FASHION_PREFIX, default_model          # noqa: E402
from pierrot.models.qwen3vl.detection import draw_detections, parse_detections  # noqa: E402
from pierrot.models.qwen3vl.weights import load_pretrained                    # noqa: E402

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_PARAMS = DEFAULTS["qwen3vl"]

# 기본 모델은 infer/defaults.py 의 pretrained (없으면 학습 산출물 final 디렉토리).
_DEFAULT_MODEL = default_model("qwen3vl")


# ------------------------------------------------------------------ #
# 추론 엔트리포인트.
# 모델/프로세서 로드 → 이미지·프롬프트 인코딩 → generate → 새 토큰만 디코드·출력.
# --model 기본값은 infer/defaults.py 의 pretrained(또는 학습 산출물 final).
# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL 추론")
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help="HF Hub id 또는 로컬 경로(파인튜닝 결과 포함). 기본값은 infer/defaults.py 의 pretrained")
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--prompt", default=None,
                        help="프롬프트(프리픽스). 미지정 시 기본 설명 프롬프트, --fashion 이면 학습 지시문")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="미지정 시 200 (--fashion 이면 JSON 절단 방지를 위해 1024)")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)  # 공식 generation_config 기본
    parser.add_argument("--top-p", type=float, default=0.8)        # 공식 기본
    parser.add_argument("--top-k", type=int, default=20)           # 공식 기본(0=비활성)
    parser.add_argument("--dtype", default=_PARAMS["dtype"], choices=list(_DTYPES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detect", action="store_true", help="출력을 검출(텍스트 좌표)로 파싱")
    parser.add_argument("--fashion", action="store_true",
                        help="출력을 패션 속성 JSON 으로 파싱하고 roi(0~999)를 픽셀 좌표로 복원")
    parser.add_argument("--save-viz", default=None, help="검출 박스를 그려 저장할 경로")
    # 전처리(동적 해상도) 옵션 — 미지정(None)이면 모델 sidecar(파인튜닝 산출물) → args → 공식값 순으로 상속.
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="이미지 총 픽셀 상한(≈토큰수×1024). 미지정 시 모델 sidecar/기본값 상속")
    parser.add_argument("--min-pixels", type=int, default=None,
                        help="이미지 총 픽셀 하한. 미지정 시 모델 sidecar/기본값 상속")
    args = parser.parse_args()

    # 프롬프트/토큰 상한 기본값 — 패션 모드는 학습 지시문과 동일해야 형식이 재현된다
    # (infer/defaults.py 의 FASHION_PREFIX 가 단일 소스).
    if args.prompt is None:
        args.prompt = FASHION_PREFIX if args.fashion else "Describe this image."
    if args.max_new_tokens is None:
        args.max_new_tokens = 1024 if args.fashion else 200

    dtype = _DTYPES[args.dtype]
    # ★ 전처리 설정 우선순위(다운로드 후 build_processor 안에서 판정 — 로컬/Hub 동일):
    #   ① CLI 명시 → ② 파인튜닝 모델의 sidecar(자체완결) → ③ 공식 base 모델(sidecar 없음)이면
    #      infer/defaults.py 안전 기본값(fallback) → ④ 공식 preprocessor_config.json.
    #   학습과 다른 max_pixels 로 추론하면 이미지 토큰 수가 달라져 성능이 떨어지므로,
    #   CLI 미지정은 None 으로 넘겨 sidecar 에 위임한다.
    model, processor = load_pretrained(
        args.model, device=args.device, dtype=dtype,
        min_pixels=args.min_pixels,                      # None=미지정(sidecar 위임)
        max_pixels=args.max_pixels,                      # None=미지정
        fallback_proc_kwargs={                            # sidecar 없는 base 모델에만 적용
            "min_pixels":    _PARAMS.get("min_pixels"),
            "max_pixels":    _PARAMS.get("max_pixels"),
            "system_prompt": _PARAMS.get("system_prompt"),
        },
    )
    model.eval()

    image  = Image.open(args.image).convert("RGB")
    inputs = processor(images=[image], text=[args.prompt])
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    generated = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"].to(dtype),
        image_grid_thw=inputs["image_grid_thw"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_token_id=processor.eos_token_id,
    )

    # generate 는 전체 시퀀스를 반환한다 — 프롬프트 뒤 새 토큰만 디코드.
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[0, prompt_len:]
    text       = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(text)

    # 패션 모드: 생성된 JSON 을 파싱해 roi(0~999 정규 좌표)를 원본 픽셀로 복원 후 정형 출력.
    if args.fashion:
        try:
            label = json.loads(text)
        except json.JSONDecodeError:
            print("\n[fashion] JSON 파싱 실패 — 위 원문을 그대로 사용하세요")
        else:
            w, h = image.size
            for items in label.values():
                if not isinstance(items, list):
                    continue
                for it in items:
                    roi = it.get("roi") if isinstance(it, dict) else None
                    if isinstance(roi, list) and len(roi) == 4:
                        it["roi"] = [round(roi[0] / 1000 * w), round(roi[1] / 1000 * h),
                                     round(roi[2] / 1000 * w), round(roi[3] / 1000 * h)]
            print("\n[fashion] roi 픽셀 복원 결과:")
            print(json.dumps(label, ensure_ascii=False, indent=2))

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
