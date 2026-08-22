"""Qwen3-VL 검출 출력 파싱 & 시각화.

모델이 생성한 검출 문자열
    "{class}: x0,y0,x1,y1 ; {class}: x0,y0,x1,y1 ; ..."
을 원본 이미지 픽셀 좌표의 박스로 되돌리고(0~999 정규 정수 → 픽셀), 이미지에 그린다.
(Qwen 토크나이저엔 <loc> 특수토큰이 없어 평문 숫자 좌표를 쓴다 — dataset.bbox_to_text
의 역변환. 좌표가 원본 크기 기준이라 동적 리사이즈 격자와 무관하게 복원된다.)
"""

from __future__ import annotations

import re
from typing import Dict, List

from PIL import Image, ImageDraw

_NUMS_RE = re.compile(r"(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)")


# ------------------------------------------------------------------ #
# 검출 문자열을 파싱해 [{box:[x0,y0,x1,y1] 픽셀, label}] 리스트로 반환한다.
# 각 세그먼트("class: x0,y0,x1,y1")에서 좌표 4개(0~999)를 찾아 원본 W/H 로 역정규화하고,
# 좌표 앞의 텍스트를 라벨로 삼는다. 좌표가 없는 조각은 건너뛴다.
# ------------------------------------------------------------------ #
def parse_detections(text: str, img_w: int, img_h: int) -> List[Dict]:
    results: List[Dict] = []
    for seg in text.split(";"):
        m = _NUMS_RE.search(seg)
        if not m:
            continue
        x0, y0, x1, y1 = (int(v) / 1000.0 for v in m.groups())
        label = seg[:m.start()].strip().rstrip(":").strip()
        results.append({
            "box":   [x0 * img_w, y0 * img_h, x1 * img_w, y1 * img_h],  # xyxy 픽셀
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
    for det in detections:
        x0, y0, x1, y1 = det["box"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        label = det["label"]
        if label:
            ty = max(0, y0 - 12)
            draw.rectangle([x0, ty, x0 + 7 * len(label) + 4, ty + 12], fill=color)
            draw.text((x0 + 2, ty), label, fill="white")
    img.save(out_path)
