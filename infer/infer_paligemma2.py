#!/usr/bin/env python
"""PaliGemma2 추론 엔트리포인트.

가중치(--model)는 세 가지를 모두 받는다:
    · 미지정      : infer/defaults.py 의 output_dir 에서 최신 checkpoint-<step> 자동 선택
                    (없으면 final/, 그것도 없으면 pretrained)
    · 디렉토리    : 학습 중 체크포인트(accelerate save_state) 또는 학습 산출물 final/
    · HF Hub id   : 예) google/paligemma2-3b-pt-896

사용:
    # 최신 체크포인트로 여러 장 검출 추론 + 시각화 저장(<원본이름>_pred.jpg)
    python infer/infer_paligemma2.py --images a.jpg b.jpg --prompt "detect shoes" --detect --save-dir preds
    # 특정 체크포인트 지정
    python infer/infer_paligemma2.py --model outputs/paligemma2_ft/checkpoint-8000 --images a.jpg
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import torch
from PIL import Image

# 이 스크립트는 infer/ 하위에 있으므로, 최상위의 pierrot 패키지를 import 하려면
# repo 루트를 sys.path 에 넣어야 한다(직접 실행 시 script 디렉토리만 잡히므로).
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from infer.defaults import DEFAULTS                                          # noqa: E402
from pierrot.models.paligemma2.detection import (                             # noqa: E402
    draw_comparison, draw_detections, parse_detections,
)
from pierrot.models.paligemma2.weights import load_from_checkpoint, load_pretrained  # noqa: E402

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_PARAMS = DEFAULTS["paligemma2"]


# ------------------------------------------------------------------ #
# print 출력을 화면과 로그 파일에 동시에 흘려보낸다(스텝별 로그 저장용).
# ------------------------------------------------------------------ #
class _Tee:
    def __init__(self, log_path: str):
        self._term = sys.__stdout__
        # 줄 버퍼링(buffering=1)으로 열어, 실행 중에도 로그가 실시간으로 디스크에
        # 쌓이게 한다. 기본 버퍼링이면 프로세스가 끝날 때까지 파일이 비어 보이고,
        # 중간에 kill 되면 내용이 통째로 유실된다.
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, s: str):
        self._term.write(s)
        self._file.write(s)
        self._file.flush()          # 매 write 마다 flush → 실시간 반영·kill 안전

    def flush(self):
        self._term.flush()
        self._file.flush()


# ------------------------------------------------------------------ #
# 사용할 가중치 경로를 정한다.
# 우선순위: --model > output_dir 의 최신 checkpoint-<step> > final/ > args.pretrained
# ------------------------------------------------------------------ #
def resolve_model_path(explicit, output_dir: str, fallback):
    if explicit:
        return explicit
    found = []
    for c in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", c)
        if m and os.path.isdir(c):
            found.append((int(m.group(1)), c))
    if found:
        return max(found)[1]
    final = os.path.join(output_dir, "final")
    return final if os.path.isdir(final) else fallback


# ------------------------------------------------------------------ #
# 결과 파일명에 붙일 가중치 태그를 만든다.
#   checkpoint-8000 → "step8000" / final → "final" / HF id → 마지막 경로명
# 어떤 체크포인트로 뽑은 결과인지 파일명만 봐도 알 수 있게 한다.
# ------------------------------------------------------------------ #
def ckpt_tag(path: str) -> str:
    m = re.search(r"checkpoint-(\d+)", str(path))
    if m:
        return f"step{m.group(1)}"
    base = os.path.basename(str(path).rstrip("/"))
    return re.sub(r"[^0-9A-Za-z._-]", "_", base) or "model"


# ------------------------------------------------------------------ #
# 경로 종류에 맞게 (model, processor) 를 로드한다.
# accelerate 체크포인트(config.json 없음)면 base 에서 구조/토크나이저를 가져오고
# 가중치만 덮어쓴다. 그 외(HF id / final 디렉토리)는 일반 로더 사용.
# ------------------------------------------------------------------ #
def load_model(path: str, base, device: str, dtype):
    is_accel_ckpt = os.path.isdir(path) and not os.path.exists(os.path.join(path, "config.json"))
    if is_accel_ckpt:
        return load_from_checkpoint(path, base=base, device=device, dtype=dtype)
    return load_pretrained(path, device=device, dtype=dtype)


# ------------------------------------------------------------------ #
# 생성 토큰과 그 확률을 ';' 구간(검출 1개)별로 묶어 파싱하고 신뢰도를 붙인다.
# 별도 confidence head 가 없으므로 서로 다른 신호를 섞지 않고 따로 보존한다:
#   score       = 첫 생성 위치의 전체 <loc> 확률 질량(현재 checkpoint 진단용 존재 신호)
#   class_score = 클래스명 토큰들의 기하평균(프롬프트 조건부라 존재확률이 아님)
#   loc_score   = <loc> 좌표 토큰 4개의 기하평균(좌표 bin 확신도)
# 좌표는 1024개 bin 중 하나를 고르는 문제라 인접 bin 에 확률이 분산되어 본질적으로
# 낮게 나온다(실측 ~0.15). 그래서 둘을 합치면 검출 신뢰도가 과소평가되므로 분리한다.
# ------------------------------------------------------------------ #
def parse_with_scores(processor, token_ids, token_probs, w: int, h: int,
                      presence_score=None):
    # 특수토큰(<eos> 등)은 라벨/점수에서 제외한다(라벨 오염·점수 왜곡 방지).
    specials = set(getattr(processor.tokenizer, "all_special_ids", []) or [])
    keep     = [(int(t), float(p)) for t, p in zip(token_ids, token_probs) if int(t) not in specials]
    pieces   = [processor.tokenizer.decode([t]) for t, _ in keep]
    probs    = [p for _, p in keep]

    dets, buf_txt, buf_p = [], [], []

    def geo_mean(vals):
        if not vals:
            return None
        return math.exp(sum(math.log(max(v, 1e-9)) for v in vals) / len(vals))

    def flush():
        if not buf_txt:
            return
        seg = "".join(buf_txt)
        # 좌표 토큰과 클래스명 토큰의 확률을 나눠 담는다
        loc_p = [p for t, p in zip(buf_txt, buf_p) if t.startswith("<loc")]
        cls_p = [p for t, p in zip(buf_txt, buf_p) if not t.startswith("<loc")]
        for d in parse_detections(seg, w, h):
            # 서로 의미가 다른 척도를 fallback 으로 섞지 않는다.
            d["score"]       = presence_score
            d["class_score"] = geo_mean(cls_p)
            d["loc_score"]   = geo_mean(loc_p)
            dets.append(d)

    for piece, p in zip(pieces, probs):
        if ";" in piece:                       # 검출 구분자 → 현재 구간 마감
            flush()
            buf_txt, buf_p = [], []
            continue
        buf_txt.append(piece)
        buf_p.append(p)
    flush()
    return dets


# ------------------------------------------------------------------ #
# 전체쿼리(all 모드) 출력을 클래스별로 쪼갠다.
#   생성 텍스트 "<loc..> shoes ; <loc..> tops ; ..." 를 ';' 로 나눠, 각 조각의
#   끝 클래스명으로 묶는다. → 개별쿼리(per_class)와 같은 "detect X → ..." 줄을 만든다.
#   반환: {클래스명: ["<loc..> 클래스", ...]}
# ------------------------------------------------------------------ #
def split_output_by_class(text: str, class_names):
    from collections import defaultdict
    groups = defaultdict(list)
    for seg in text.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        # 긴 이름부터 맞춰 "scarves & muffler" 가 "muffler" 로 오인되지 않게 한다
        for c in sorted(class_names, key=len, reverse=True):
            if seg.endswith(c):
                groups[c].append(seg)
                break
    return groups


# 액세서리 클래스 — 몸에 걸치는 작은 소품들.
# 이들은 박스가 작아 좌표 오차가 상대적으로 크고, 없는데도 그럴듯한 위치를
# 만들어내기 쉬워서 loc(좌표 확신도) 하한을 추가로 요구한다.
# bags·shoes·hats 는 제외한다 — 박스가 크고 경계가 뚜렷해 loc 이 낮게 나오는 일이
# 잦은데(가방 0.06~0.12) 실제로는 존재하는 경우가 많아 관문에 걸리면 손해다.
ACCESSORY_CLASSES = {
    "earrings", "necklaces", "bracelets", "rings", "brooches", "hair accessory",
    "eyewear", "ties", "scarves & muffler", "watch", "gloves", "belt", "tights",
}


# ------------------------------------------------------------------ #
# 검출을 두 조건 중 "하나라도" 만족하면 통과시킨다.
#   ① conf(존재 신호) == 1               → cls 와 무관하게 통과
#   ② conf 와 cls(클래스 확률)가 동시에 0.999 이상
# 즉 conf 가 1 이면 그것만으로 충분하고, 1 에 못 미칠 때만 cls 가 보조 판정을 한다.
# conf 는 1024개 loc 확률의 합이라 부동소수점상 정확히 1.0 이 되는 일은 드물다.
# 그래서 ① 은 "소수 3자리 표시가 1.000" (즉 0.9995 이상) 으로 판정한다.
#
# 액세서리(ACCESSORY_CLASSES)는 위 조건을 통과해도 loc >= acc_loc_th 를 추가로 요구한다.
# 작은 소품은 conf/cls 가 쉽게 포화되지만 없는 물건을 지어낼 때 좌표가 흔들려
# loc 이 낮게 나오므로, loc 이 실질적인 존재 판정 기준이 된다.
# 점수가 없는(None) 검출은 판정 근거가 없으므로 탈락으로 본다.
#   반환: (통과 리스트, 탈락 리스트) — 탈락분은 로그로 보여 왜 빠졌는지 알 수 있게 한다.
# ------------------------------------------------------------------ #
def filter_by_scores(detections, conf_th: float, cls_th: float, acc_loc_th: float = 0.14):
    kept, dropped = [], []
    for d in detections:
        conf = d.get("score")
        cls  = d.get("class_score")
        if conf is None or cls is None:
            dropped.append(d)
            continue
        # ① conf 가 1 (표시 기준 1.000) → 단독으로 통과
        is_one = round(conf, 3) >= 1.0
        # ② conf·cls 동시에 하한 이상
        both   = conf >= conf_th and cls >= cls_th
        ok     = is_one or both
        # 액세서리 추가 관문: 좌표를 충분히 확신해야 인정한다
        if ok and d["label"] in ACCESSORY_CLASSES:
            loc = d.get("loc_score")
            if loc is None or loc < acc_loc_th:
                ok = False
                # 왜 떨어졌는지 로그에서 구분되도록 사유를 남긴다
                d["drop_reason"] = f"액세서리 loc<{acc_loc_th}"
        (kept if ok else dropped).append(d)
    return kept, dropped


# ------------------------------------------------------------------ #
# 두 박스가 "서로" 얼마나 덮는지 계산한다(양방향 최소값).
#   교집합/A면적, 교집합/B면적 중 작은 값 → 둘 다 이 비율 이상 겹친다는 뜻.
# IoU 대신 이 값을 쓰는 이유: 큰 박스가 작은 박스를 품는 경우 IoU 는 낮게 나오지만
# 양방향 비율은 작은 쪽 기준으로 1.0 에 가까워, "같은 자리"를 더 잘 잡아낸다.
# ------------------------------------------------------------------ #
def mutual_overlap(box_a, box_b) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    # 교집합 사각형
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter    = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a   = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b   = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return min(inter / area_a, inter / area_b)


# ------------------------------------------------------------------ #
# 서로 90% 이상 겹치는 검출들을 같은 객체로 보고 점수 최고 1개만 남긴다.
# per_class 추론은 클래스마다 따로 물어보므로 클래스간 경쟁이 없다. 그래서
# 같은 바지에 pants 와 skirts 가 동시에 잡히는 중복이 생긴다 → 여기서 정리한다.
# 승자는 2단계로 고른다:
#   1순위 conf(존재 신호) — 더 높은 쪽이 바로 이긴다.
#   2순위 conf 가 같으면 세 신호의 합(conf + cls + loc)으로 가른다.
# conf 는 로그에 소수 4자리로 찍히므로 같은 자리에서 비교해야 표시와 판정이 일치한다.
# (원본 float 로 비교하면 0.99991 vs 0.99990 처럼 눈에 안 보이는 차이로 승부가 갈려
#  로그만 봐서는 왜 그렇게 됐는지 알 수 없다)
# 클래스 구분 없이(class-agnostic) 비교한다 — 중복의 원인이 클래스간 충돌이기 때문.
#   반환: (남긴 리스트, 흡수된 리스트) — 흡수분도 로그로 보여준다.
# ------------------------------------------------------------------ #
CONF_DECIMALS = 4


def dedup_overlapping(detections, overlap_th: float = 0.80):
    # 세 점수의 합(없는 값은 0)
    def total(d) -> float:
        return sum(d.get(k) or 0.0 for k in ("score", "class_score", "loc_score"))

    # 정렬 키: (표시 자리수로 맞춘 conf, 점수합) — 앞이 같을 때만 뒤를 본다
    def rank(d):
        return (round(d.get("score") or 0.0, CONF_DECIMALS), total(d))

    kept, merged = [], []
    # 점수 높은 것부터 자리를 차지하고, 이미 찬 자리와 겹치면 흡수시킨다
    for d in sorted(detections, key=rank, reverse=True):
        # 이미 자리를 차지한 것들 중 임계값을 넘게 겹치는 첫 번째
        hit = next(((k, ov) for k in kept
                    if (ov := mutual_overlap(d["box"], k["box"])) >= overlap_th), None)
        if hit is None:
            kept.append(d)
        else:
            # 어느 검출에 얼마나 겹쳐서 밀렸는지 남겨 로그에서 추적 가능하게 한다
            d["merged_into"] = hit[0]["label"]
            d["overlap"]     = hit[1]
            merged.append(d)
    return kept, merged


# ------------------------------------------------------------------ #
# 최종 검출들의 모든 쌍 겹침 비율을 (비율 내림차순) 문자열 목록으로 만든다.
# 임계값을 못 넘어 살아남은 겹침이 얼마나 되는지 눈으로 확인하기 위한 것.
# ------------------------------------------------------------------ #
def overlap_pairs(detections):
    pairs = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            ov = mutual_overlap(detections[i]["box"], detections[j]["box"])
            if ov > 0:
                pairs.append((ov, detections[i]["label"], detections[j]["label"]))
    return sorted(pairs, reverse=True)


# ------------------------------------------------------------------ #
# COCO split 디렉토리에서 GT 를 읽어 {파일명: [{box(xyxy), label}]} 로 만든다.
# 예측과 나란히 비교/시각화하기 위한 것(없으면 GT 없이 예측만 표시).
# ------------------------------------------------------------------ #
def load_gt(coco_dir: str):
    import json
    with open(os.path.join(coco_dir, "_annotations.coco.json"), encoding="utf-8") as f:
        coco = json.load(f)
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    gt = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        gt.setdefault(id2file[ann["image_id"]], []).append(
            {"box": [x, y, x + w, y + h], "label": id2name.get(ann["category_id"], "object")}
        )
    return gt


# ------------------------------------------------------------------ #
# 추론 엔트리포인트.
# 모델/프로세서를 1회 로드 → 이미지들을 순회하며 generate → 새 토큰만 디코드 출력.
# 검출 모드면 loc 토큰을 박스로 파싱하고, 이미지별 이름으로 시각화를 저장한다.
# ------------------------------------------------------------------ #
def main() -> None:
    p = argparse.ArgumentParser(description="PaliGemma2 추론")
    p.add_argument("--model", default=None, help="가중치 경로/HF id (미지정 시 최신 checkpoint 자동)")
    p.add_argument("--images", nargs="+", required=True, help="입력 이미지 경로(여러 개 가능)")
    p.add_argument("--prompt", default="caption en", help="프롬프트(프리픽스)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--dtype", default=_PARAMS.get("dtype", "bfloat16"), choices=list(_DTYPES))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--detect", action="store_true", help="출력을 검출(loc 토큰)로 파싱")
    p.add_argument("--save-dir", default="results/paligemma2",
                   help="검출 시각화 저장 디렉토리(<원본이름>_pred.jpg). 알고리즘별 results/<모델>")
    p.add_argument("--coco-dir", default=None,
                   help="GT 비교용 COCO split 디렉토리(_annotations.coco.json). 지정 시 GT+예측 함께 표시")
    p.add_argument("--classes", nargs="*", default=None,
                   help="클래스별로 'detect {클래스}' 를 각각 실행해 결과를 합친다(per_class 학습과 일치). "
                        "'gt' 를 주면 각 이미지의 GT 클래스들을 사용")
    p.add_argument("--conf-th", type=float, default=0.6,
                   help="존재 신호(conf) 하한. 이 값 미만이면 그리지 않는다. "
                        "강제복원 박스는 conf=loc_mass 라 presence-loc-mass 와 같은 값(0.6)으로 맞춰야 "
                        "복원분이 필터에서 다시 잘리지 않는다")
    p.add_argument("--cls-th", type=float, default=0.75,
                   help="클래스 확률(cls) 하한. 이 값 미만이면 그리지 않는다")
    p.add_argument("--presence-margin", type=float, default=None,
                   help="greedy 가 EOS 를 골라도 loc_mass-eos 가 이 값보다 크면 박스를 강제한다"
                        "(좌표 흩어짐으로 큰 객체가 통째로 누락되는 것 방지, 예: 0.5). None=끔")
    p.add_argument("--presence-loc-mass", type=float, default=0.6,
                   help="greedy 가 EOS 를 골라도 loc_mass(<loc> 확률 합)가 이 값 이상이면 박스를 강제한다"
                        "(절대 기준, 기본 0.6). presence-margin 과 OR 로 결합. None/0 이하=끔")
    p.add_argument("--acc-loc-th", type=float, default=0.14,
                   help="액세서리 클래스에만 적용하는 loc(좌표 확신도) 하한")
    p.add_argument("--overlap-th", type=float, default=0.80,
                   help="서로 이 비율 이상 겹치면 같은 객체로 보고 점수(conf+cls+loc) 최고 1개만 남긴다")
    a = p.parse_args()

    path  = resolve_model_path(a.model, _PARAMS.get("output_dir", "./outputs"), _PARAMS.get("pretrained"))
    dtype = _DTYPES[a.dtype]
    # 결과 파일명에 붙일 체크포인트 태그(예: step8000)
    tag   = ckpt_tag(path)

    # 로그·결과를 스텝별로 남긴다: save_dir/<tag>/ 아래에 이미지와 infer_<tag>.log 저장.
    # 스텝(체크포인트)마다 폴더가 갈려 이전 결과를 덮어쓰지 않는다.
    if a.save_dir:
        a.save_dir = os.path.join(a.save_dir, tag)
        os.makedirs(a.save_dir, exist_ok=True)
        # 이후 모든 print 를 화면과 스텝별 로그 파일에 동시에 기록(tee).
        sys.stdout = _Tee(os.path.join(a.save_dir, f"infer_{tag}.log"))

    print(f"[infer] weights={path} ({tag}) | device={a.device} | dtype={a.dtype}")

    model, processor = load_model(path, _PARAMS.get("pretrained"), a.device, dtype)
    model.eval()
    # 첫 생성 위치에서 합산할 1024개 loc 토큰 id. processor 가 초기화 때 모두 보장한다.
    loc_ids = [processor.tokenizer.convert_tokens_to_ids(f"<loc{i:04d}>") for i in range(1024)]
    if len(set(loc_ids)) != 1024:
        raise RuntimeError("토크나이저의 <loc0000..1023> id 가 고유하지 않아 presence 신호를 계산할 수 없습니다.")
    loc_token_ids = torch.tensor(
        loc_ids,
        dtype=torch.long,
        device=a.device,
    )

    gt_map = load_gt(a.coco_dir) if a.coco_dir else {}

    for img_path in a.images:
        image  = Image.open(img_path).convert("RGB")
        w, h   = image.size
        gts    = gt_map.get(os.path.basename(img_path), [])
        print(f"\n[{os.path.basename(img_path)}]")

        # 실행할 프롬프트 목록을 정한다.
        #   --classes 지정 → 클래스마다 "detect {클래스}" 한 번씩 (per_class 학습과 일치)
        #   'gt' 지정      → 그 이미지의 GT 클래스들을 사용
        #   미지정         → --prompt 하나만
        if a.classes is not None:
            names   = sorted({g["label"] for g in gts}) if a.classes == ["gt"] else a.classes
            prompts = [f"detect {c}" for c in names]
        else:
            prompts = [a.prompt]

        detections = []
        for prompt in prompts:
            inputs = processor(images=[image], text=[prompt])
            inputs = {k: v.to(a.device) for k, v in inputs.items()}
            generated, probs, signals = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(dtype),
                attention_mask=inputs["attention_mask"],
                max_new_tokens=a.max_new_tokens,
                do_sample=a.do_sample,
                temperature=a.temperature,
                top_p=a.top_p,
                eos_token_id=processor.eos_token_id,
                return_scores=True,
                presence_token_ids=loc_token_ids,
                presence_margin=a.presence_margin,
                presence_loc_mass=a.presence_loc_mass,
            )
            # 프롬프트 뒤 생성 토큰만 디코드(loc 토큰은 special 이 아니라 보존됨).
            new_tokens = generated[0, inputs["input_ids"].shape[-1]:]
            text       = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            loc_mass = float(signals["loc_mass"][0])
            eos_prob = float(signals["eos_prob"][0])
            # presence 강제로 EOS 를 뚫고 살아난 생성이면 로그에 [강제복원] 표시.
            forced   = bool(signals["forced"][0]) if "forced" in signals else False
            forced_tag = "  [강제복원]" if forced else ""

            # 로그 출력: 전체쿼리(all 모드)도 개별쿼리(per_class)와 같은 형식으로 찍는다.
            #   개별쿼리 → 프롬프트가 클래스 1개라 그대로 "detect X → ..." 한 줄.
            #   전체쿼리 → 한 번의 생성 결과를 클래스별로 쪼개 "detect X → ..." 줄들로 편다.
            #   (per-class loc_mass/eos 신호는 클래스별 생성이 없어 전체쿼리엔 1줄만 나온다)
            is_all_mode = a.classes is None and ";" in prompt
            if is_all_mode:
                class_names = [c.strip() for c in prompt.replace("detect", "", 1).split(";")]
                groups = split_output_by_class(text, class_names)
                for c in class_names:
                    print(f"  · 'detect {c}' → {' ; '.join(groups.get(c, []))}")
                print(f"    signal(전체쿼리) loc_mass={loc_mass:.6f} eos={eos_prob:.6f} "
                      f"margin={loc_mass - eos_prob:+.6f}{forced_tag}")
            else:
                print(f"  · {prompt!r} → {text}")
                print(f"    signal loc_mass={loc_mass:.6f} eos={eos_prob:.6f} "
                      f"margin={loc_mass - eos_prob:+.6f}{forced_tag}")
            # 토큰 확률을 검출별(';' 로 나뉜 구간)로 묶어 신뢰도로 붙인다.
            dets = parse_with_scores(
                processor, new_tokens, probs[0], w, h, presence_score=loc_mass
            )
            # 강제복원된 검출은 개별 항목에도 표시(dedup/필터 뒤에도 추적 가능하게).
            for d in dets:
                d["forced"] = forced
            detections += dets

        # ① 점수 하한 통과 → ② 서로 크게 겹치는 중복 제거, 순서로 정리한다.
        detections, dropped = filter_by_scores(detections, a.conf_th, a.cls_th, a.acc_loc_th)
        detections, merged  = dedup_overlapping(detections, a.overlap_th)

        if a.detect or a.save_dir:

            # 점수별 표시 문자열(통과/탈락 공통).
            def _line(tag: str, d) -> str:
                x0, y0, x1, y1 = (round(v, 1) for v in d["box"])
                # conf/cls 는 임계값(0.999)이 4번째 자리에서 갈리므로 4자리로 보여준다.
                # (3자리면 통과/탈락이 같은 값으로 보여 판정 근거를 알 수 없다)
                conf = f"  conf={d['score']:.4f}" if d.get("score") is not None else ""
                cls  = f" cls={d['class_score']:.4f}" if d.get("class_score") is not None else ""
                loc  = f" loc={d['loc_score']:.3f}" if d.get("loc_score") is not None else ""
                frc  = " [강제복원]" if d.get("forced") else ""
                return f"    {tag} {d['label']:18s} [{x0}, {y0}, {x1}, {y1}]{conf}{cls}{loc}{frc}"

            # GT 와 예측을 나란히 출력(감지 결과 비교).
            if gts:
                for g in gts:
                    x0, y0, x1, y1 = (round(v, 1) for v in g["box"])
                    print(f"    GT   {g['label']:18s} [{x0}, {y0}, {x1}, {y1}]")
            for d in detections:
                print(_line("PRED", d))
            # 중복으로 흡수된 검출(어느 쪽에 얼마나 겹쳐서 밀렸는지 함께 표시).
            for d in merged:
                print(_line("mrge", d) + f"  → {d['merged_into']} (겹침 {d['overlap']:.3f})")
            # 임계값에 걸려 제외된 검출(왜 안 그려졌는지 확인용).
            for d in dropped:
                why = f"  ({d['drop_reason']})" if d.get("drop_reason") else ""
                print(_line("drop", d) + why)
            # 남은 검출들끼리 실제로 얼마나 겹치는지 전부 표시(임계값 미만이라 남은 것들).
            for ov, la, lb in overlap_pairs(detections):
                mark = " ←임계값 근접" if ov >= a.overlap_th * 0.9 else ""
                print(f"    ovlp {la:14s} ~ {lb:14s} {ov:.3f}{mark}")
            if gts and detections:
                print(f"    IoU  {_best_iou(detections, gts):.3f} (예측-정답 최대 겹침)")
            if not detections:
                print("    PRED (없음)")

            if a.save_dir:
                # 입력 파일명 + 체크포인트 태그로 저장. GT 있으면 함께 그린다.
                stem = os.path.splitext(os.path.basename(img_path))[0]
                out  = os.path.join(a.save_dir, f"{stem}_pred_{tag}.jpg")
                if gts:
                    draw_comparison(image, detections, gts, out)
                else:
                    draw_detections(image, detections, out)
                print(f"    → 저장: {out}  (초록=GT, 빨강=PRED)" if gts else f"    → 저장: {out}")


# ------------------------------------------------------------------ #
# 예측·정답 박스 쌍 중 최대 IoU 를 구한다(간단한 정합도 지표).
# ------------------------------------------------------------------ #
def _best_iou(preds, gts) -> float:
    def iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih   = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter    = iw * ih
        ua       = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0
    return max((iou(p["box"], g["box"]) for p in preds for g in gts), default=0.0)


if __name__ == "__main__":
    main()
