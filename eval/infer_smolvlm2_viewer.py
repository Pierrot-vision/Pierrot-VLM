#!/usr/bin/env python
"""SmolVLM2 자유형 추론(캡션/VQA) → 큰 폰트 단일 HTML 뷰어.

학습 산출물(final)이나 HF id 로 여러 이미지×프롬프트를 생성해 이미지·질문·답을 한 페이지에
담는다. 이미지는 data URI 로 임베드해 외부 파일 없이 단일 HTML 로 열람/다운로드 가능.

사용:
    python eval/infer_smolvlm2_viewer.py \
        --model outputs/smolvlm2_finevision/final --glob 'data/det_test/*.jpg' --limit 6
"""
from __future__ import annotations

import os

import argparse, base64, glob, io, sys
from contextlib import nullcontext

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pierrot.models.smolvlm2.weights import load_pretrained

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
PROMPTS = ["Describe this image in detail.", "What is the main object and its color?"]


def _thumb(image: Image.Image, max_side: int = 420) -> str:
    im = image.convert("RGB"); w, h = im.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        im = im.resize((int(w * s), int(h * s)))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="SmolVLM2 추론 뷰어")
    ap.add_argument("--model", default="outputs/smolvlm2_finevision/final")
    ap.add_argument("--glob", default="data/det_test/*.jpg")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--dtype", default="float32", choices=list(_DTYPES))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/smolvlm2/infer_viewer.html")
    a = ap.parse_args()

    model, proc = load_pretrained(a.model, device=a.device, dtype=_DTYPES[a.dtype]); model.eval()
    paths = sorted(glob.glob(a.glob))[:a.limit]
    print(f"[infer] model={a.model} imgs={len(paths)} split={proc.do_image_splitting} "
          f"max_splits={proc.max_splits_per_side} size_le={proc.size_longest_edge}")
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(a.device).startswith("cuda") else nullcontext()

    cards = ""
    for p in paths:
        im = Image.open(p).convert("RGB")
        qa = ""
        for pr in PROMPTS:
            enc = proc(images=[im], text=[pr]); enc = {k: v.to(a.device) for k, v in enc.items()}
            with amp:
                gen = model.generate(input_ids=enc["input_ids"], pixel_values=enc["pixel_values"].to(_DTYPES[a.dtype]),
                                     attention_mask=enc["attention_mask"], max_new_tokens=a.max_new_tokens,
                                     do_sample=False, eos_token_id=proc.eos_token_id)
            out = proc.tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            print(f"  {os.path.basename(p)} | {pr} -> {out[:80]}...")
            qa += (f"<div style='font-size:21px;margin:10px 0'><b style='color:#2563eb'>Q:</b> {esc(pr)}<br>"
                   f"<b style='color:#1b7f37'>A:</b> {esc(out)}</div>")
        cards += (f"<div style='border:1px solid #ddd;border-radius:12px;padding:16px;margin:16px 0;overflow:auto'>"
                  f"<div style='font-size:16px;color:#888'>{esc(os.path.basename(p))} ({im.size[0]}×{im.size[1]})</div>"
                  f"<img src='{_thumb(im)}' style='max-width:420px;max-height:420px;border-radius:8px;"
                  f"float:left;margin:8px 20px 8px 0'>{qa}</div>")

    html = (f"<!doctype html><meta charset='utf-8'><title>SmolVLM2 추론</title>"
            f"<div style='font-family:sans-serif;max-width:1100px;margin:24px auto;padding:0 16px'>"
            f"<h1 style='font-size:32px'>SmolVLM2 추론 샘플 — {esc(os.path.basename(a.model.rstrip('/')))}</h1>"
            f"<div style='font-size:18px;color:#666'>이미지 {len(paths)}장 · greedy · max_new_tokens={a.max_new_tokens}</div>{cards}</div>")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[saved] {a.out}")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
