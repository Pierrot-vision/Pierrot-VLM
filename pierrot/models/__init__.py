"""모델 패키지 — 이 배포본에는 다섯 알고리즘의 추론 경로만 들어 있다.

레지스트리 등록(spec)이 없으므로 여기서 하위 패키지를 import 하지 않는다.
필요한 모델만 직접 import 하면 그 모델의 의존만 끌어온다:

    from pierrot.models.qwen3vl import load_pretrained
"""

__all__ = ["paligemma2", "nanovlm", "smolvlm2", "qwen3vl", "qwen35"]
