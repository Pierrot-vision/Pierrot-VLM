"""nanoVLM 알고리즘 패키지 (추론 전용 배포본).

학습 배포본과 달리 레지스트리 등록(spec)·데이터셋 어댑터가 없다. 체크포인트를 읽는
load_pretrained() 와 생성 경로만 노출한다.

무거운 의존(transformers/einops/torchvision)은 실제 로드 시점에 끌어온다.
"""

from .config import VLMConfig
from .modeling.vision_language_model import VisionLanguageModel
from .processor import NanoVLMProcessor
from .weights import load_pretrained

__all__ = [
    "VLMConfig",
    "VisionLanguageModel",
    "NanoVLMProcessor",
    "load_pretrained",
]
