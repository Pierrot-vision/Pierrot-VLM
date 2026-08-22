#!/usr/bin/env python
"""SmolVLM2 진짜 벤치마크 eval — MMStar 객관식 정확도(자체 구현, lmms-eval 불필요).

nanoVLM 의 eval/eval_nanovlm_bench.py 와 동일한 프로토콜을, SmolVLM2 모델 API
(pixel_values / generate(do_sample=False)) 에 맞춘 것이다.

무엇을 재나:
  - train/eval loss 는 '학습 건강 프록시'일 뿐(정답 여러개면 낮게 안 나옴).
  - 이 스크립트는 '진짜 능력 수치' — 이미지 보고 정답 글자를 골라 맞힌 비율(정확도).

두 모드:
  - likelihood(권장): 각 선택지 글자의 로그확률을 비교(장황한 모델도 공정). 문항당 forward 1회.
  - generate: 글자를 실제 생성(원본 프로토콜).

사용:
    # 최신 산출물(args/smolvlm2.py 의 output_dir/final) 로 MMStar 전체
    python eval/eval_smolvlm2_bench.py
    # 표본 제한(스모크) / 특정 체크포인트
    python eval/eval_smolvlm2_bench.py --limit 200 --ckpt outputs/smolvlm2_finevision/final
    # 공개 base 모델과 비교(구현 검증)
    python eval/eval_smolvlm2_bench.py --ckpt HuggingFaceTB/SmolVLM2-2.2B-Instruct
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import argparse
import base64
import glob
import io
import json
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from typing import List, Tuple

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer.defaults import DEFAULTS                             # 체크포인트 경로 단일 소스
from pierrot.models.smolvlm2.weights import load_pretrained

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


# ------------------------------------------------------------------ #
# 체크포인트 로드. final/(config.json+model.pt) 또는 HF id → load_pretrained.
# 전처리 설정은 산출물의 smolvlm2_preprocessor.json(sidecar)에서 자동 복원된다.
# ------------------------------------------------------------------ #
def load_for_eval(ckpt: str, device: str, dtype, max_splits=None):
    # max_splits 명시 시 CLI 우선(sidecar/공식값 무시) — 해상도 통제 실험용.
    return load_pretrained(ckpt, device=device, dtype=dtype, max_splits_per_side=max_splits)


# ------------------------------------------------------------------ #
# encode_one 결과 → 모델 입력 텐서 (input_ids, attention_mask, pixel_values(1,n,3,H,W)).
# ------------------------------------------------------------------ #
def _to_inputs(enc, device):
    input_ids = torch.tensor([enc["input_ids"]], device=device)
    attn      = torch.tensor([enc["attention_mask"]], device=device)
    pixels    = torch.stack(enc["pixel_values"]).unsqueeze(0).to(device)   # (1, n_tiles, 3, H, W)
    return input_ids, attn, pixels


# ------------------------------------------------------------------ #
# MMStar 질문 정규화(원본 nanoVLM 포맷과 동일). 선택지 블록을 "Choices:" 로.
# ------------------------------------------------------------------ #
def format_question(q: str) -> str:
    q = q.replace("\nOptions:", "\nChoices:")
    q = q.replace("\nA. ", "\nChoices:\nA. ")
    q = q.replace("Please select the correct answer from the options above.", "Answer with the letter.")
    q = q.replace("Answer with the option's letter from the given choices directly", "Answer with the letter directly")
    return q


# ------------------------------------------------------------------ #
# 생성 텍스트 → 예측 글자(A~E). 우선순위 패턴으로 '실제 답 글자'만 추출.
# ------------------------------------------------------------------ #
def parse_letter(text: str) -> str:
    t = text.strip()
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([A-E])\)?\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.match(r"\(?([A-E])\)?(?:[\.\):,]|\s|$)", t)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:option|choice)\s*\(?([A-E])\)?\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([B-E])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\bA\b(?!\s+[a-z])", t)
    return "A" if m else ""


# ------------------------------------------------------------------ #
# 질문에 실제로 존재하는 선택지 글자 목록(예: A~D). 없으면 A~D.
# ------------------------------------------------------------------ #
def option_letters(question: str) -> List[str]:
    present = [L for L in "ABCDE" if re.search(rf"(?:^|\n)\s*{L}[\.\):]", question)]
    return present or ["A", "B", "C", "D"]


# ------------------------------------------------------------------ #
# 우도 기반 예측: 프롬프트 마지막 위치 logits(첫 답토큰 분포)에서 각 선택지 글자의
# 첫 답토큰 logprob 을 비교해 argmax.
# ------------------------------------------------------------------ #
@torch.no_grad()
def predict_likelihood(model, processor, image: Image.Image, question: str, device: str) -> Tuple[str, str]:
    img   = image.convert("RGB")
    q     = format_question(question)
    enc_p = processor.encode_one(img, q, suffix=None)
    input_ids, attn, pixels = _to_inputs(enc_p, device)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    with amp:
        out = model(input_ids=input_ids, pixel_values=pixels, attention_mask=attn)   # {"logits"}
    logprobs = torch.log_softmax(out["logits"][0, -1].float(), dim=-1)

    best_letter, best_score = "", -1e30
    for L in option_letters(question):
        enc_l  = processor.encode_one(img, q, suffix=L)
        labels = enc_l["labels"]
        first  = next((i for i, v in enumerate(labels) if v != -100), None)
        if first is None:
            continue
        tok = enc_l["input_ids"][first]
        s   = logprobs[tok].item()
        if s > best_score:
            best_score, best_letter = s, L
    return best_letter, f"loglik(best={best_letter}:{best_score:.2f})"


# ------------------------------------------------------------------ #
# 한 문항 greedy 생성 → 텍스트.
# ------------------------------------------------------------------ #
@torch.no_grad()
def predict_one(model, processor, image: Image.Image, question: str, device: str, max_new_tokens: int) -> str:
    prompt = format_question(question) + "\nAnswer:"
    enc = processor.encode_one(image.convert("RGB"), prompt, suffix=None)
    input_ids, attn, pixels = _to_inputs(enc, device)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    with amp:
        gen = model.generate(
            input_ids=input_ids, pixel_values=pixels, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=processor.eos_token_id,
        )
    new = gen[0, input_ids.shape[1]:]
    return processor.tokenizer.decode(new, skip_special_tokens=True)


# ------------------------------------------------------------------ #
# 최신 accelerate 체크포인트(checkpoint-N 중 N 최대). 없으면 final.
# ------------------------------------------------------------------ #
def latest_checkpoint(output_dir: str) -> str:
    final = os.path.join(output_dir, "final")
    if os.path.isdir(final) and os.path.exists(os.path.join(final, "config.json")):
        return final
    cks = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if cks:
        return max(cks, key=lambda p: int(p.rsplit("-", 1)[-1]))
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description="SmolVLM2 MMStar 벤치마크")
    ap.add_argument("--ckpt", default=None, help="체크포인트 dir / HF id (기본: output_dir/final)")
    ap.add_argument("--hf-id", default="Lin-Chen/MMStar", help="벤치 HF repo")
    ap.add_argument("--split", default="val", help="분할(MMStar val=1500)")
    ap.add_argument("--limit", type=int, default=None, help="표본 수 제한(스모크용)")
    ap.add_argument("--mode", default="likelihood", choices=["likelihood", "generate"])
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--max-splits-per-side", type=int, default=None, help="타일 상한 강제(미지정=sidecar/공식값)")
    ap.add_argument("--dtype", default="float32", choices=list(_DTYPES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-view", type=int, default=16, help="HTML 뷰어에 담을 표본 수")
    ap.add_argument("--out-dir", default="results/smolvlm2", help="결과 저장 위치")
    a = ap.parse_args()

    ckpt = a.ckpt or latest_checkpoint(DEFAULTS["smolvlm2"]["output_dir"])
    print(f"[eval] ckpt={ckpt} | device={a.device} dtype={a.dtype} mode={a.mode}")
    model, processor = load_for_eval(ckpt, a.device, _DTYPES[a.dtype], max_splits=a.max_splits_per_side)
    model.eval()

    from datasets import load_dataset
    ds = load_dataset(a.hf_id, split=a.split)
    n_total = len(ds) if a.limit is None else min(a.limit, len(ds))
    print(f"[eval] {a.hf_id}/{a.split}: {len(ds)}문항 중 {n_total} 평가")

    correct = 0
    per_cat = defaultdict(lambda: [0, 0])
    samples = []
    for i in range(n_total):
        row = ds[i]
        q   = row["question"]
        gt  = str(row["answer"]).strip().upper()[:1]
        cat = row.get("category") or row.get("l2_category") or "?"
        if a.mode == "likelihood":
            pred, text = predict_likelihood(model, processor, row["image"], q, a.device)
        else:
            text = predict_one(model, processor, row["image"], q, a.device, a.max_new_tokens)
            pred = parse_letter(text)
        ok = (pred == gt)
        correct += ok
        per_cat[cat][0] += ok
        per_cat[cat][1] += 1
        if len(samples) < a.n_view:
            samples.append({"idx": i, "q": q, "gt": gt, "pred": pred, "raw": text,
                            "cat": cat, "ok": bool(ok), "thumb": _thumb(row["image"])})
        if (i + 1) % 100 == 0 or i + 1 == n_total:
            print(f"  {i + 1}/{n_total}  acc={correct / (i + 1):.4f}")

    acc = correct / max(n_total, 1)
    cat_acc = {c: v[0] / v[1] for c, v in sorted(per_cat.items())}
    print(f"\n===== MMStar 정확도: {acc:.4f}  ({correct}/{n_total}) =====")
    for c, v in cat_acc.items():
        print(f"  {c:24} {v:.4f}  ({per_cat[c][0]}/{per_cat[c][1]})")

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{os.path.basename(ckpt.rstrip('/'))}_{a.mode}"
    result = {"ckpt": ckpt, "mode": a.mode, "n": n_total, "accuracy": acc, "per_category": cat_acc}
    with open(os.path.join(a.out_dir, f"bench_mmstar_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _write_viewer(os.path.join(a.out_dir, f"bench_mmstar_{tag}.html"), tag, acc, n_total, cat_acc, samples)
    print(f"\n[saved] {a.out_dir}/bench_mmstar_{tag}.json / .html")


# ------------------------------------------------------------------ #
# PIL → 축소 JPEG data URI(단일 HTML 임베드).
# ------------------------------------------------------------------ #
def _thumb(image: Image.Image, max_side: int = 384) -> str:
    im = image.convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ #
# 큰 폰트 단일 HTML 뷰어 — 정확도/카테고리표 + 표본별 이미지·Q/GT/Pred(정오 색).
# ------------------------------------------------------------------ #
def _write_viewer(path, tag, acc, n, cat_acc, samples) -> None:
    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    rows = "".join(
        f"<tr><td>{esc(c)}</td><td style='text-align:right'>{v:.3f}</td></tr>" for c, v in cat_acc.items()
    )
    cards = ""
    for s in samples:
        color = "#1b7f37" if s["ok"] else "#c0392b"
        mark  = "✅" if s["ok"] else "❌"
        img   = (f"<img src='{s['thumb']}' style='max-width:360px;max-height:360px;border-radius:8px;"
                 f"float:left;margin:0 16px 8px 0'>" if s.get("thumb") else "")
        cards += (
            f"<div style='border:2px solid {color};border-radius:10px;padding:14px;margin:12px 0;overflow:auto'>"
            f"<div style='font-size:18px;color:#555'>[{s['idx']}] {esc(s['cat'])} {mark}</div>"
            f"{img}"
            f"<div style='font-size:22px;margin:8px 0;white-space:pre-wrap'>{esc(s['q'])}</div>"
            f"<div style='font-size:24px'>GT=<b>{esc(s['gt'])}</b> &nbsp; "
            f"Pred=<b style='color:{color}'>{esc(s['pred'] or '∅')}</b> "
            f"<span style='color:#888;font-size:18px'>(raw: {esc(s['raw'])})</span></div></div>"
        )
    html = (
        f"<!doctype html><meta charset='utf-8'><title>MMStar {esc(tag)}</title>"
        f"<div style='font-family:sans-serif;max-width:1000px;margin:24px auto;padding:0 16px'>"
        f"<h1 style='font-size:34px'>MMStar — {esc(tag)}</h1>"
        f"<div style='font-size:40px;font-weight:700'>정확도 {acc:.4f} "
        f"<span style='font-size:22px;color:#666'>(n={n})</span></div>"
        f"<table style='font-size:20px;border-collapse:collapse;margin:16px 0'>"
        f"<tr><th style='text-align:left'>category</th><th>acc</th></tr>{rows}</table>"
        f"<h2 style='font-size:26px'>표본 {len(samples)}개</h2>{cards}</div>"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
