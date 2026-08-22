"""추론 기본값 단일 소스 — 모델별 체크포인트·dtype·전처리 예산.

학습 저장소(Pierrot-VLM-Lab)의 `args/<모델>.py` PARAMS 에서 **추론에 영향을 주는 키만**
뽑아 온 것이다. 학습 하이퍼파라미터(lr·배치·스텝·데이터 경로)는 여기에 없다.

★ 여기 한 곳만 고치면 해당 모델 추론 스크립트의 기본값이 바뀐다.

왜 전처리 값이 추론에도 중요한가: 이미지 토큰 수를 정하는 손잡이(qwen 계열의
max_pixels, smolvlm2 의 타일 분할)가 학습과 달라지면 시퀀스 구성이 달라져 품질이
떨어진다. 파인튜닝 산출물에는 이 값들이 sidecar(<모델>_preprocessor.json)로 동봉되어
자동 상속되므로, 아래 값은 **sidecar 가 없는 공개 base 체크포인트용 fallback** 이다.
"""

from __future__ import annotations

DEFAULTS: dict = {

    # ── PaliGemma2 (SigLIP-So400m + Gemma2) ─────────────────────────────
    'paligemma2': {
        'pretrained' : 'google/paligemma2-3b-pt-896',   # 896 해상도 베이스
        'dtype'      : 'bfloat16',
        'output_dir' : './outputs/paligemma2_ft',       # --model 미지정 시 최신 체크포인트를 여기서 찾는다
    },

    # ── nanoVLM (SigLIP2 ViT + 픽셀셔플 + SmolLM2) ──────────────────────
    'nanovlm': {
        'pretrained' : 'lusxvr/nanoVLM-450M',           # 공개 base. 우리 산출물은 --model 로 지정
        # nanoVLM 정석은 FP32 파라미터 + bf16 autocast 라 학습·평가 모두 float32 를 쓴다.
        # 순수 bf16 로 돌리면 RoPE(cos/sin=fp32)가 q,k 를 승격시켜 SDPA 에서 dtype 이 어긋난다.
        'dtype'      : 'float32',
        'output_dir' : './outputs/nanovlm_mix',
    },

    # ── SmolVLM2 (SigLIP + 픽셀셔플 커넥터 + SmolLM2, Idefics3 타일링) ──
    'smolvlm2': {
        'pretrained'          : 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
        # nanoVLM 과 같은 이유로 float32 (학습 설정과 동일). 순수 bf16 은 어텐션 마스크와
        # 쿼리의 dtype 이 어긋나 SDPA 가 죽는다 — 메모리가 빠듯하면 --dtype float16 을 쓸 것.
        'dtype'               : 'float32',
        'output_dir'          : './outputs/smolvlm2_finevision',
        # 타일 분할 — 작은 물체에 유리하지만 타일 수만큼 시퀀스가 길어진다.
        'do_image_splitting'  : True,
        'size_longest_edge'   : None,                   # None=체크포인트 공식값(256M/500M=2048, 2.2B=1536)
        'max_splits_per_side' : 2,                      # 변당 최대 타일 수(최악 5타일)
    },

    # ── Qwen3-VL (동적해상도 ViT + DeepStack + Qwen3) ───────────────────
    'qwen3vl': {
        'pretrained'    : 'Qwen/Qwen3-VL-2B-Instruct',
        'dtype'         : 'bfloat16',
        'output_dir'    : './outputs/qwen3vl_fashion',
        # 이미지 토큰 수 ≈ max_pixels / 1024 (1024 = (patch16×merge2)² = 32²).
        # 320*32*32 = 327,680 → 이미지당 최대 320 토큰. 작은 물체면 640*32*32 로 키운다.
        'min_pixels'    : 256 * 256,
        'max_pixels'    : 320 * 32 * 32,
        'system_prompt' : None,                          # 공식 템플릿 기본은 system 턴 없음
    },

    # ── Qwen3.5 (동적해상도 ViT + Gated DeltaNet 하이브리드 디코더) ─────
    'qwen35': {
        'pretrained'      : 'Qwen/Qwen3.5-4B',
        'dtype'           : 'bfloat16',
        'output_dir'      : './outputs/qwen35_fashion',
        'min_pixels'      : 256 * 256,
        'max_pixels'      : 320 * 32 * 32,
        'system_prompt'   : None,
        # thinking 템플릿 모드. False = 빈 사고 블록 뒤 바로 답변(검출·속성 파인튜닝 기본).
        # 파인튜닝 산출물은 sidecar 값을 상속하므로 여기 값은 base 체크포인트용이다.
        'enable_thinking' : False,
    },
}

# 패션 속성 태스크(Qwen3-VL/Qwen3.5 파인튜닝)의 지시문 — 학습 데이터 빌더가 모든 샘플에
# 똑같이 넣은 문장이다. 추론에서도 **글자 하나까지 같아야** 출력 형식이 재현된다.
FASHION_PREFIX = 'Analyze every fashion item in the image and return their attributes as JSON.'


# ------------------------------------------------------------------ #
# --model 미지정 시 쓸 기본 경로: 공개 pretrained, 없으면 학습 산출물 final/.
# ------------------------------------------------------------------ #
def default_model(name: str) -> str:
    p = DEFAULTS[name]
    return p.get('pretrained') or f"{p.get('output_dir', './outputs')}/final"
