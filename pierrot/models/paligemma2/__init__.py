"""PaliGemma2 알고리즘 패키지 (추론 전용 배포본).

학습 배포본과 달리 레지스트리 등록(spec)·데이터셋 어댑터가 없다. 체크포인트를 읽는
load_pretrained() / load_from_checkpoint() 와 생성 경로만 노출한다.
"""

from .config import Gemma2Config, PaliGemma2Config, SiglipConfig
from .modeling.paligemma2 import PaliGemma2ForConditionalGeneration
from .processor import PaliGemma2Processor
from .weights import build_model_from_config, load_from_checkpoint, load_pretrained

__all__ = [
    "PaliGemma2Config",
    "Gemma2Config",
    "SiglipConfig",
    "PaliGemma2ForConditionalGeneration",
    "PaliGemma2Processor",
    "load_pretrained",
    "load_from_checkpoint",
    "build_model_from_config",
]
