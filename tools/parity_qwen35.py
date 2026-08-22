#!/usr/bin/env python
"""Qwen3.5 스크래치 구현 vs HF 공식 구현 수치 parity 검증 (opt-in).

기본 테스트(tests/)는 다운로드 불필요 원칙이라 실제 4B 가중치 비교를 포함하지 않는다.
이 스크립트가 그 간극을 메운다 — Gated DeltaNet/게이트 어텐션/RoPE/전처리를 손대면
이걸 돌려 회귀를 잡는다.

요구사항(충족 못 하면 이유를 출력하고 skip 종료):
  - transformers >= 5.1 (Qwen3_5ForConditionalGeneration 지원)
  - Qwen/Qwen3.5-4B 다운로드 가능(HF Hub 또는 로컬 캐시)
  - RAM: 두 모델을 **순차** 로드(공식 → 해제 → 커스텀)하므로 fp32 기준 peak 약 20~24GB.

실행:
    python tools/parity_qwen35.py                       # 기본(4B, CPU, fp32)
    python tools/parity_qwen35.py --model <id/경로> --device cuda

검증 항목(모두 통과해야 exit 0):
  ① 전처리: input_ids 완전 동일(thinking 접두 포함) · image_grid_thw 동일 ·
     pixel_values max|diff| ≤ 2/255
  ② 비전 타워(동일 픽셀 입력): 머저 출력 max|diff| ≤ 1e-3
  ③ 전체 forward: 로짓 argmax 일치율 = 100%
  ④ greedy 생성: 토큰 id 시퀀스 완전 동일(허용 예외는 말미 EOS 1토큰 차이뿐)
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

MODEL_DEFAULT = "Qwen/Qwen3.5-4B"
MIN_PIXELS    = 256 * 256
MAX_PIXELS    = 512 * 32 * 32          # 작은 예산으로 빠르게(비교 목적엔 충분)
PROMPT        = "Describe this image in one sentence."


# ------------------------------------------------------------------ #
# 결정론적 합성 이미지(파일/네트워크 의존 없음): 그라데이션 + 빨간 사각형.
# ------------------------------------------------------------------ #
def _make_image() -> Image.Image:
    arr = np.zeros((300, 420, 3), np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 420, dtype=np.uint8)[None, :]
    arr[:, :, 1] = np.linspace(0, 255, 300, dtype=np.uint8)[:, None]
    arr[80:200, 120:300] = (240, 30, 30)
    return Image.fromarray(arr, "RGB")


# ------------------------------------------------------------------ #
# 요구사항 검사: transformers 버전이 Qwen3.5 를 모르면 skip(실패 아님) 종료.
# ------------------------------------------------------------------ #
def _require_hf_qwen35():
    import transformers
    try:
        from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
    except ImportError:
        print(f"[skip] transformers {transformers.__version__} 는 Qwen3.5 를 지원하지 않습니다 "
              f"(>=5.1 필요). parity 는 별도 환경에서 실행하세요.")
        sys.exit(0)
    return transformers.__version__


# ------------------------------------------------------------------ #
# 1단계: 공식 HF 구현으로 기준값을 모두 계산해 CPU 텐서로 반환한 뒤 모델을 해제한다.
# (두 4B 모델을 동시에 들고 있지 않기 위한 순차 실행 — peak RAM 절반)
# thinking 접두는 우리 프로세서 기본과 같은 non-thinking 으로 고정한다.
# ------------------------------------------------------------------ #
def _run_official(model_id: str, device: str, image: Image.Image, max_new_tokens: int) -> dict:
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    proc  = AutoProcessor.from_pretrained(model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float32).to(device).eval()

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text   = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        vis    = model.model.visual(inputs["pixel_values"].float(), grid_thw=inputs["image_grid_thw"])
        merged = vis[0] if isinstance(vis, tuple) else vis.pooler_output
        logits = model(**inputs).logits.float()
        gen    = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    ref = {
        "input_ids":      inputs["input_ids"].cpu(),
        "attention_mask": inputs["attention_mask"].cpu(),
        "image_grid_thw": inputs["image_grid_thw"].cpu(),
        "pixel_values":   inputs["pixel_values"].float().cpu(),
        "vision_merged":  merged.cpu(),
        "logits":         logits.cpu(),
        "gen_new":        gen[0, inputs["input_ids"].shape[1]:].cpu().tolist(),
        "decode":         lambda ids: proc.tokenizer.decode(ids, skip_special_tokens=True),
    }
    # 공식 모델 해제(커스텀 모델 로드 전 RAM 회수).
    del model, inputs, vis, merged, logits, gen
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 parity 검증(opt-in)")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    ver = _require_hf_qwen35()
    print(f"transformers {ver} | model {args.model} | device {args.device}")

    image = _make_image()
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # ---------- 1단계: 공식 기준값 계산 후 해제 ----------
    print("\n== 공식 HF 기준값 계산(완료 후 모델 해제) ==")
    ref = _run_official(args.model, args.device, image, args.max_new_tokens)
    print(f"  기준값 확보: ids {tuple(ref['input_ids'].shape)}, grid {ref['image_grid_thw'].tolist()}")

    # ---------- 2단계: 커스텀 구현 로드 ----------
    from pierrot.models.qwen35.weights import load_pretrained
    our_model, our_proc = load_pretrained(
        args.model, device=args.device, dtype=torch.float32,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, enable_thinking=False)
    our_model.eval()
    our_in = {k: v.to(args.device) for k, v in our_proc(images=[image], text=[PROMPT]).items()}

    # ---------- ① 전처리 (thinking 접두 포함 input_ids 동일해야 함) ----------
    print("\n== ① 전처리 ==")
    check("input_ids 동일", torch.equal(ref["input_ids"], our_in["input_ids"].cpu()),
          f"HF {tuple(ref['input_ids'].shape)} vs ours {tuple(our_in['input_ids'].shape)}")
    check("image_grid_thw 동일", torch.equal(ref["image_grid_thw"], our_in["image_grid_thw"].cpu()),
          f"{ref['image_grid_thw'].tolist()} vs {our_in['image_grid_thw'].tolist()}")
    pv_hf, pv_our = ref["pixel_values"], our_in["pixel_values"].float().cpu()
    pix_ok = pv_hf.shape == pv_our.shape and float((pv_hf - pv_our).abs().max()) <= 2 / 255
    check("pixel_values ≤ 2/255", pix_ok,
          f"max|diff|={float((pv_hf - pv_our).abs().max()):.6f}" if pv_hf.shape == pv_our.shape
          else f"shape {tuple(pv_hf.shape)} vs {tuple(pv_our.shape)}")

    # ---------- ② 비전 타워(공식과 동일한 픽셀 입력으로) ----------
    print("\n== ② 비전 타워 ==")
    with torch.no_grad():
        our_merged = our_model.model.visual(
            pv_hf.to(args.device), ref["image_grid_thw"].to(args.device))
    d = float((ref["vision_merged"] - our_merged.cpu()).abs().max())
    check("머저 출력 ≤ 1e-3", d <= 1e-3, f"max|diff|={d:.6f}")

    # ---------- ③ 전체 forward 로짓 ----------
    print("\n== ③ 전체 forward ==")
    with torch.no_grad():
        our_logits = our_model(
            input_ids=ref["input_ids"].to(args.device), pixel_values=pv_hf.to(args.device),
            image_grid_thw=ref["image_grid_thw"].to(args.device),
            attention_mask=ref["attention_mask"].to(args.device))["logits"].float().cpu()
    agree = float((ref["logits"].argmax(-1) == our_logits.argmax(-1)).float().mean())
    check("로짓 argmax 100% 일치", agree == 1.0,
          f"일치율={agree:.4f}, max|diff|={float((ref['logits'] - our_logits).abs().max()):.5f}")

    # ---------- ④ greedy 생성 ----------
    print("\n== ④ greedy 생성 ==")
    with torch.no_grad():
        our_gen = our_model.generate(
            input_ids=our_in["input_ids"], pixel_values=our_in["pixel_values"],
            image_grid_thw=our_in["image_grid_thw"], attention_mask=our_in["attention_mask"],
            max_new_tokens=args.max_new_tokens, eos_token_id=our_proc.eos_token_id)
    hf_new  = ref["gen_new"]
    our_new = our_gen[0, our_in["input_ids"].shape[1]:].cpu().tolist()

    # 완전 동일이 기준. 허용하는 유일한 예외는 "말미 EOS 1토큰 차이"
    # (EOS 를 출력에 포함/제외하는 구현 관례 차이일 뿐 내용은 같음).
    eos_ids = set(our_proc.eos_token_id if isinstance(our_proc.eos_token_id, (list, tuple))
                  else [our_proc.eos_token_id])
    same = hf_new == our_new
    if not same and abs(len(hf_new) - len(our_new)) == 1:
        n      = min(len(hf_new), len(our_new))
        longer = hf_new if len(hf_new) > len(our_new) else our_new
        same   = n > 0 and hf_new[:n] == our_new[:n] and longer[-1] in eos_ids
    check("greedy 토큰 시퀀스 동일(±말미 EOS 1개)", same,
          f"HF={len(hf_new)}tok, ours={len(our_new)}tok")
    print("  HF :", ref["decode"](hf_new))
    print("  our:", our_proc.tokenizer.decode(our_new, skip_special_tokens=True))

    # ---------- 결과 ----------
    print(f"\n{'PASS — 전 항목 parity 일치' if not failures else 'FAIL — ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
