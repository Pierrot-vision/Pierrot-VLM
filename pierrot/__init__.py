"""Pierrot-VLM-Infer: 스크래치 구현 VLM **추론** 패키지 (OCR 제외 태스크).

[Pierrot-VLM-Lab](https://github.com/Pierrot-vision/Pierrot-VLM-Lab) 에서 학습·재구현한 다섯
모델(PaliGemma2 · nanoVLM · SmolVLM2 · Qwen3-VL · Qwen3.5)을 돌리는 데 필요한 부분만
떼어낸 배포본이다. 학습 엔진(Accelerate 루프)·데이터 빌더·데이터셋 어댑터는 들어 있지
않다 — 체크포인트를 읽어 생성하는 경로만 있다.

문서 파싱(OCR) 추론은 별도 배포본 [Pierrot-VLM-OCR](https://github.com/Pierrot-vision/Pierrot-VLM-OCR)
에 있다.

    from pierrot.models.qwen3vl import load_pretrained
    model, processor = load_pretrained("Qwen/Qwen3-VL-2B-Instruct", device="cuda")
"""

__all__ = ["models"]

__version__ = "0.1.0"
