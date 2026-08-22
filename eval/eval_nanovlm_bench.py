#!/usr/bin/env python
"""nanoVLM 진짜 벤치마크 eval — MMStar 객관식 정확도(자체 구현, lmms-eval 불필요).

왜 자체 구현: 원본 nanoVLM 은 무거운 lmms-eval 프레임워크(+수많은 태스크 의존/네트워크)를
쓴다. MMStar 는 1500문항 단순 객관식(A~E) + 정답 내장이라, 프롬프트 포맷만 원본과 맞추면
로컬에서 정확도를 바로 계산할 수 있다. 학습 중간중간 체크포인트 능력을 재는 용도.

무엇을 재나:
  - 학습 루프의 held-out aokvqa token-acc 는 '학습 건강 프록시'일 뿐(정답 여러개면 낮게 나옴).
  - 이 스크립트는 '진짜 능력 수치' — 이미지 보고 정답 글자를 골라 맞힌 비율(정확도).

프롬프트 포맷: 원본 nanoVLM eval/lmms_eval_wrapper.py 의 (ai2d,mmstar,seedbench,scienceqa)
그룹 규칙을 따른다 — "Options→Choices" 정규화 + assistant 를 "Answer:" 로 프라이밍.

사용:
    # 최신 체크포인트(outputs/nanovlm_mix) 로 MMStar 전체
    python eval/eval_nanovlm_bench.py
    # 특정 체크포인트 / 표본 수 제한(스모크)
    python eval/eval_nanovlm_bench.py --ckpt outputs/nanovlm_mix/checkpoint-500 --limit 50
    # 공개 모델과 비교(구현/데이터 검증)
    python eval/eval_nanovlm_bench.py --ckpt lusxvr/nanoVLM-450M
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

import torch
from PIL import Image

# repo 루트를 path 에 추가(eval/ 하위 직접 실행 대비)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer.defaults import DEFAULTS                             # 체크포인트 경로 단일 소스
from pierrot.models.nanovlm.weights import load_pretrained

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_LETTER = re.compile(r"[A-E]")                  # 생성 텍스트에서 첫 선택지 글자 추출


# ------------------------------------------------------------------ #
# 체크포인트 로드. 두 종류를 모두 지원:
#   config.json 이 있는 디렉토리(학습 산출물 final) 또는 HF id 를 받는다.
#
# 학습 중간의 accelerate 체크포인트(checkpoint-N: model.safetensors 만 있고 config 가 없음)는
# 아키텍처를 args 로 세워야 해서 이 배포본에서는 지원하지 않는다 — 학습 저장소에서
# final 로 저장한 뒤(또는 config.json 을 함께 둔 뒤) 넣으면 된다.
# ------------------------------------------------------------------ #
def load_for_eval(ckpt: str, device: str, dtype):
    if os.path.isdir(ckpt) and not os.path.exists(os.path.join(ckpt, "config.json")):
        raise SystemExit(
            f"[eval] {ckpt} 에 config.json 이 없습니다 — 학습 중간 체크포인트(accelerate save_state)는 "
            f"이 추론 배포본에서 로드할 수 없습니다. 학습 산출물 final/ 또는 HF id 를 지정하세요."
        )
    return load_pretrained(ckpt, device=device, dtype=dtype)


# ------------------------------------------------------------------ #
# MMStar 질문 텍스트 정규화(원본 nanoVLM 포맷과 동일). 선택지 블록을 "Choices:" 로,
# 지시문을 "Answer with the letter." 로 바꿔 모델이 글자만 답하도록 유도.
# ------------------------------------------------------------------ #
def format_question(q: str) -> str:
    q = q.replace("\nOptions:", "\nChoices:")
    q = q.replace("\nA. ", "\nChoices:\nA. ")
    q = q.replace("Please select the correct answer from the options above.", "Answer with the letter.")
    q = q.replace("Answer with the option's letter from the given choices directly", "Answer with the letter directly")
    return q


# ------------------------------------------------------------------ #
# 생성 텍스트 → 예측 글자(A~E). 단순히 첫 A~E 를 잡으면 "The" 의 E 처럼 단어 속
# 글자를 오인식하므로, 우선순위 패턴으로 '실제 답 글자'만 추출한다.
#   ① "Answer: B" / "answer is B"  ② 맨 앞 "B" "B." "B)" "(B)"  ③ "option/choice B"
#   ④ 최후: 독립 단어 글자(관사 'A' 오인식 줄이려 B~E 우선)
# ------------------------------------------------------------------ #
def parse_letter(text: str) -> str:
    t = text.strip()
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([A-E])\)?\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.match(r"\(?([A-E])\)?(?:[\.\):,]|\s|$)", t)          # 맨 앞 글자+구두점/공백/끝
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:option|choice)\s*\(?([A-E])\)?\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([B-E])\b", t)                            # 독립 글자(B~E 먼저: 관사 A 회피)
    if m:
        return m.group(1)
    m = re.search(r"\bA\b(?!\s+[a-z])", t)                      # 'A' 는 뒤에 소문자단어(관사) 아니면
    return "A" if m else ""


# ------------------------------------------------------------------ #
# 질문에 실제로 존재하는 선택지 글자 목록(예: A~D). "A. " "B) " 등을 스캔. 없으면 A~D.
# ------------------------------------------------------------------ #
def option_letters(question: str) -> List[str]:
    present = [L for L in "ABCDE" if re.search(rf"(?:^|\n)\s*{L}[\.\):]", question)]
    return present or ["A", "B", "C", "D"]


# ------------------------------------------------------------------ #
# 우도(likelihood) 기반 예측: 모델이 각 선택지 글자에 부여하는 로그확률을 비교해 argmax.
# 생성이 장황해도(글자를 안 뱉어도) '어느 답을 더 그럴듯하게 보는가'를 직접 잰다.
#   - 프롬프트(assistant 시작까지)를 1회 forward → 마지막 위치 logits = 첫 답토큰 예측분포.
#   - 각 글자의 '첫 답토큰 id'(학습과 동일한 encode 경로로 획득)의 logprob 을 비교.
# ------------------------------------------------------------------ #
@torch.no_grad()
def predict_likelihood(model, processor, image: Image.Image, question: str, device: str) -> Tuple[str, str]:
    img   = image.convert("RGB")
    q     = format_question(question)
    enc_p = processor.encode_one(img, q, suffix=None)           # 프롬프트(=공통 접두)
    Lp    = len(enc_p["input_ids"])
    input_ids      = torch.tensor([enc_p["input_ids"]], device=device)
    attention_mask = torch.tensor([enc_p["attention_mask"]], device=device)
    images         = [enc_p["images"].to(device)]
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    with amp:
        out = model(input_ids=input_ids, images=images, attention_mask=attention_mask)   # labels 없음 → {"logits"}
    logprobs = torch.log_softmax(out["logits"][0, -1].float(), dim=-1)   # (V,) 첫 답토큰 분포

    best_letter, best_score = "", -1e30
    for L in option_letters(question):
        enc_l  = processor.encode_one(img, q, suffix=L)         # 학습과 동일 인코딩으로 글자 토큰 확보
        labels = enc_l["labels"]
        first  = next((i for i, v in enumerate(labels) if v != -100), None)
        if first is None:
            continue
        tok = enc_l["input_ids"][first]                         # 그 글자의 '첫 답토큰' id
        s   = logprobs[tok].item()
        if s > best_score:
            best_score, best_letter = s, L
    return best_letter, f"loglik(best={best_letter}:{best_score:.2f})"


# ------------------------------------------------------------------ #
# 한 문항 추론: 이미지+정규화 질문을 인코딩 → greedy 생성 → 글자 예측.
# ------------------------------------------------------------------ #
@torch.no_grad()
def predict_one(model, processor, image: Image.Image, question: str, device: str, max_new_tokens: int) -> str:
    prompt = format_question(question) + "\nAnswer:"        # assistant 프라이밍(원본 포맷)
    enc = processor.encode_one(image.convert("RGB"), prompt, suffix=None)
    input_ids      = torch.tensor([enc["input_ids"]], device=device)
    attention_mask = torch.tensor([enc["attention_mask"]], device=device)
    images         = [enc["images"].to(device)]
    # ★ 학습과 동일한 계산경로: fp32 가중치 + bf16 autocast. autocast 없이 순수 bf16 로 돌리면
    #   RoPE(cos/sin=fp32)가 q,k 를 fp32 로 승격시켜 SDPA 가 v(bf16)와 dtype 불일치로 죽는다.
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    with amp:
        gen = model.generate(
            input_ids=input_ids, images=images, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, greedy=True, eos_token_id=processor.eos_token_id,
        )
    return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


# ------------------------------------------------------------------ #
# 최신 accelerate 체크포인트 경로(checkpoint-N 중 N 최대). 없으면 final.
# ------------------------------------------------------------------ #
def latest_checkpoint(output_dir: str) -> str:
    cks = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if cks:
        return max(cks, key=lambda p: int(p.rsplit("-", 1)[-1]))
    return os.path.join(output_dir, "final")


def main() -> None:
    ap = argparse.ArgumentParser(description="nanoVLM MMStar 벤치마크")
    ap.add_argument("--ckpt", default=None, help="체크포인트 dir / HF id (기본: outputs 최신 checkpoint-N)")
    ap.add_argument("--task", default="mmstar", choices=["mmstar"], help="벤치마크(현재 MMStar)")
    ap.add_argument("--hf-id", default="Lin-Chen/MMStar", help="벤치 HF repo")
    ap.add_argument("--split", default="val", help="분할(MMStar 는 val=1500)")
    ap.add_argument("--limit", type=int, default=None, help="표본 수 제한(스모크용)")
    # likelihood: 각 선택지 로그확률 비교(장황한 초기 모델도 공정, 권장). generate: 글자 생성(원본 프로토콜).
    ap.add_argument("--mode", default="likelihood", choices=["likelihood", "generate"])
    ap.add_argument("--max-new-tokens", type=int, default=16)
    # 가중치는 fp32 로 두고 계산만 bf16 autocast(학습과 동일). 순수 bf16 가중치는 RoPE dtype 승격 이슈.
    ap.add_argument("--dtype", default="float32", choices=list(_DTYPES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-view", type=int, default=16, help="HTML 뷰어에 담을 표본 수")
    ap.add_argument("--out-dir", default="results/nanovlm", help="결과 저장 위치")
    a = ap.parse_args()

    ckpt = a.ckpt or latest_checkpoint(DEFAULTS["nanovlm"]["output_dir"])
    print(f"[eval] ckpt={ckpt} | task={a.task} | device={a.device} dtype={a.dtype}")
    model, processor = load_for_eval(ckpt, a.device, _DTYPES[a.dtype])

    from datasets import load_dataset
    ds = load_dataset(a.hf_id, split=a.split)
    n_total = len(ds) if a.limit is None else min(a.limit, len(ds))
    print(f"[eval] {a.hf_id}/{a.split}: {len(ds)}문항 중 {n_total} 평가")

    correct = 0
    per_cat = defaultdict(lambda: [0, 0])       # category -> [맞은수, 전체수]
    samples = []                                # HTML 뷰어용
    for i in range(n_total):
        row  = ds[i]
        q    = row["question"]
        gt   = str(row["answer"]).strip().upper()[:1]
        cat  = row.get("category") or row.get("l2_category") or "?"
        if a.mode == "likelihood":
            pred, text = predict_likelihood(model, processor, row["image"], q, a.device)
        else:
            text = predict_one(model, processor, row["image"], q, a.device, a.max_new_tokens)
            pred = parse_letter(text)
        ok   = (pred == gt)
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
    result = {"ckpt": ckpt, "task": a.task, "mode": a.mode, "n": n_total, "accuracy": acc, "per_category": cat_acc}
    with open(os.path.join(a.out_dir, f"bench_{a.task}_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _write_viewer(os.path.join(a.out_dir, f"bench_{a.task}_{tag}.html"), tag, acc, n_total, cat_acc, samples)
    print(f"\n[saved] {a.out_dir}/bench_{a.task}_{tag}.json / .html")


# ------------------------------------------------------------------ #
# PIL 이미지 → 축소 JPEG data URI(뷰어 임베드용, 외부파일 없이 단일 HTML 유지).
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
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    rows = "".join(
        f"<tr><td>{esc(c)}</td><td style='text-align:right'>{v:.3f}</td></tr>" for c, v in cat_acc.items()
    )
    cards = ""
    for s in samples:
        color = "#1b7f37" if s["ok"] else "#c0392b"
        mark  = "✅" if s["ok"] else "❌"
        img   = f"<img src='{s['thumb']}' style='max-width:360px;max-height:360px;border-radius:8px;float:left;margin:0 16px 8px 0'>" if s.get("thumb") else ""
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
