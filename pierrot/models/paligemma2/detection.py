"""PaliGemma 검출 출력 파싱 & 시각화.

모델이 생성한 검출 문자열
    "<locYMIN><locXMIN><locYMAX><locXMAX> class ; <loc...> class ; ..."
을 원본 이미지 픽셀 좌표의 박스로 되돌리고(coco.py 의 역변환), 이미지에 그린다.
"""

from __future__ import annotations

import re
from typing import Dict, List

from PIL import Image, ImageDraw

_LOC_RE = re.compile(r"<loc(\d{4})>")


# ------------------------------------------------------------------ #
# 검출 문자열을 파싱해 [{box:[x0,y0,x1,y1] 픽셀, label}] 리스트로 반환한다.
# loc 토큰은 y_min,x_min,y_max,x_max(1024 bins)이므로 원본 W/H 로 역정규화한다.
# loc 4개가 안 되는 조각(불완전 출력)은 건너뛴다.
# ------------------------------------------------------------------ #
def parse_detections(text: str, img_w: int, img_h: int) -> List[Dict]:
    results: List[Dict] = []
    for seg in text.split(";"):
        locs = _LOC_RE.findall(seg)
        if len(locs) < 4:
            continue
        ymin, xmin, ymax, xmax = (int(v) / 1024.0 for v in locs[:4])
        label = _LOC_RE.sub("", seg).strip()
        results.append({
            "box":   [xmin * img_w, ymin * img_h, xmax * img_w, ymax * img_h],  # xyxy 픽셀
            "label": label,
        })
    return results


# ------------------------------------------------------------------ #
# 파싱한 검출 결과를 이미지에 사각형+라벨로 그려 저장한다.
# ------------------------------------------------------------------ #
def draw_detections(image: Image.Image, detections: List[Dict], out_path: str,
                    color: str = "red", width: int = 3) -> None:
    img  = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    _draw_group(draw, detections, color, width, tag="", label_above=True)
    img.save(out_path)


# ------------------------------------------------------------------ #
# 정답(GT, 초록)과 예측(PRED, 빨강)을 한 이미지에 함께 그려 저장한다.
# GT 라벨은 박스 위, PRED 라벨은 박스 아래에 두어 겹치지 않게 한다.
# ------------------------------------------------------------------ #
def draw_comparison(image: Image.Image, preds: List[Dict], gts: List[Dict],
                    out_path: str, width: int = 3) -> None:
    img  = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    _draw_group(draw, gts,   "lime", width, tag="GT",   label_above=True)
    _draw_group(draw, preds, "red",  width, tag="PRED", label_above=False)
    img.save(out_path)


# ------------------------------------------------------------------ #
# 박스 묶음 하나를 그린다(사각형 + 라벨 배지).
# label_above=True 면 박스 위, False 면 박스 아래에 라벨을 붙인다.
# ------------------------------------------------------------------ #
def _draw_group(draw, dets: List[Dict], color: str, width: int,
                tag: str = "", label_above: bool = True) -> None:
    for det in dets:
        x0, y0, x1, y1 = det["box"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        # 라벨은 클래스명만 표시한다. 점수는 로그(conf/cls/loc)에서 확인한다
        # — 그림에 넣으면 값이 죄다 1.00 으로 찍혀 정보가 없고 라벨만 길어진다.
        text = f"{tag}:{det['label']}" if tag else det["label"]
        if not text:
            continue
        ty = max(0, y0 - 13) if label_above else y1 + 1
        draw.rectangle([x0, ty, x0 + 7 * len(text) + 4, ty + 12], fill=color)
        draw.text((x0 + 2, ty), text, fill="black" if color == "lime" else "white")
