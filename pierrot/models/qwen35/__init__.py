"""Qwen3.5 알고리즘 패키지 — 동적해상도 ViT + Gated DeltaNet 하이브리드 디코더 (추론 전용).

학습 배포본과 달리 레지스트리 등록(spec)·데이터셋 어댑터가 없다. 체크포인트를 읽는
load_pretrained() 와 생성 경로만 노출한다.
"""

from .modeling import Qwen35ForConditionalGeneration
from .weights import load_pretrained

__all__ = ["Qwen35ForConditionalGeneration", "load_pretrained"]
