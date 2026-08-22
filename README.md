<h1 align="center">🔮 PIERROT VLM</h1>

<p align="center">
  <b>PyTorch 스크래치 비전-언어 모델 5종 — 추론 배포본 (OCR 제외 태스크)</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="pytorch"/>
  <img src="https://img.shields.io/badge/models-5-success.svg" alt="models"/>
  <img src="https://img.shields.io/badge/inference--only-✓-brightgreen.svg" alt="inference-only"/>
  <img src="https://img.shields.io/badge/from--scratch-✓-brightgreen.svg" alt="from-scratch"/>
  <img src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-orange.svg" alt="license"/>
  <img src="https://img.shields.io/badge/commercial%20use-⛔%20prohibited-red.svg" alt="no-commercial"/>
</p>

<p align="center">
  <b>한국어</b> | <a href="README_en.md">English</a>
</p>

---

## 💡 소개

**PIERROT VLM** 는 VLM 관련 선도 알고리즘에 대한 재현 · 적용 · 성능향상 · 다양한 나의
아이디어 등을 실험하는 장소입니다. [VLM-OCR](https://github.com/Pierrot-vision/Pierrot-VLM-OCR)
은 독립적 장소에 있습니다.

> **이름의 유래** — 피에로(Pierrot)는 원래 무언극에서 **남을 따라 하고 흉내 내는** 광대
> 캐릭터입니다. 기존 연구의 좋은 점을 따라 재현·결합한다는 [Pierrot Universe](https://github.com/Pierrot-vision)
> 의 첫 번째 철학(MimiC)과 맞닿아 있어 이 이름을 씁니다.

* 현재 학습코드, 학습 모델은 공개하지 않고 있습니다.

## 📰 News

- 2026-08-22 — 🚀 **추론 코드 공개** (PaliGemma2 · nanoVLM · SmolVLM2 · Qwen3-VL · Qwen3.5)
- 2026-08-04 — 📊 **MMStar 통합 평가** 공개 (재구현 vs 공식 base)
- 2026-08-04 — 🔵 **SmolVLM2 (FineVision)** 학습 실험 결과 공개
- 2026-08-01 — 🟢 **nanoVLM (FineVision)** 학습 실험 결과 공개
- 2026-07-29 — 🟣 **Qwen3-VL (Fashion Attribute)** 실험 결과 공개
- 2026-07-28 — 🧩 **Qwen3.5-4B** 추가 (다섯 번째 모델)
- 2026-07-28 — 🎯 **PaliGemma2 (Detection)** 실험 결과 공개
- 2026-07-24 — 🟣 **Qwen3-VL** 추가 (네 번째 모델)
- 2026-07-23 — 🟩 **SmolVLM2** 추가 (세 번째 모델)
- 2026-07-22 — 🚀 **첫 공개** (PaliGemma2 · nanoVLM)

---

## 🧩 지원 모델

다섯 알고리즘 모두 **모델 코드가 순수 PyTorch 스크래치 구현**이며, 공개 체크포인트의 키와
이름을 맞춰 두어 `load_state_dict` 로 바로 들어옵니다. 모델 패키지·엔트리포인트·설정은
**서로 완전히 분리**됩니다.

| | **PaliGemma2** | **nanoVLM** | **SmolVLM2** | **Qwen3-VL** | **Qwen3.5** |
|---|---|---|---|---|---|
| 비전 인코더 | SigLIP-So400m | SigLIP2 ViT | SigLIP | 동적해상도 ViT | 동적해상도 ViT |
| 언어 모델 | Gemma 2 | SmolLM2 | SmolLM2 | Qwen3 | Qwen3.5 하이브리드 |
| 프로젝터 | [Linear](LAB/paligemma2.md#projector) | [Pixel-shuffle](LAB/nanovlm.md#projector) | [Pixel-shuffle](LAB/smolvlm2.md#projector) | [Patch merger](LAB/qwen3vl.md#projector) | [Patch merger](LAB/qwen35.md) |
| 시퀀스 | [`<bos>`+`\n`](LAB/paligemma2.md#prefix-lm) | [ChatML](LAB/nanovlm.md#chatml) | [챗 템플릿](LAB/smolvlm2.md#chat-template) | [ChatML](LAB/qwen3vl.md) | [ChatML](LAB/qwen35.md) |
| 어텐션 마스크 | [prefix-LM](LAB/paligemma2.md#prefix-lm) | 일반 causal | 일반 causal | 일반 causal | 일반 causal |
| 레이어 (비전/언어) | 27 / 26 | 12 / 32 | 27 / 24 | 24 / 28 | 24 / 32 |
| 파라미터 | ~3B | ~450M | 256M/500M/2.2B | ~2B | ~4B |
| 가중치 메모리 | ~6 GB | ~0.9 GB | ~0.5/1/4.4 GB | ~4 GB | ~8 GB |
| 이미지 토큰 | 고정 (896→4096) | 동적 타일 (64/타일) | 동적 타일 | 타일 없음 (동적) | 타일 없음 (동적) |

> **Qwen3.5** 는 **Gated DeltaNet 3:1 하이브리드 디코더**를 순수 PyTorch로 풀 스크래치 구현했습니다 (공식 체크포인트와 텐서 1:1 · 723개 검증).

**공식 구현과의 수치 대조** — Qwen3-VL · Qwen3.5 는 공개 가중치로 공식 HF 구현과 로짓·생성
토큰을 직접 맞춰 볼 수 있습니다 ([tools/parity_qwen3vl.py](tools/parity_qwen3vl.py) ·
[tools/parity_qwen35.py](tools/parity_qwen35.py)).

---

## ✅ 검증

공개 가중치로 추론 경로를 끝까지 돌려 확인했습니다.

- [x] **paligemma2**
- [x] **nanovlm**
- [x] **smolvlm2**
- [x] **qwen3vl**
- [x] **qwen35**

---

## 📊 실험 결과

학습·실험 자체는 학습 저장소에서 이뤄졌고, 아래는 그 결과를 이 배포본의 추론 경로로 재현할 수
있는 것들입니다. 자세한 설계·실패 기록은 [실험 노트(LAB)](#-실험-노트-lab) 를 참고하세요.

### 🎯 PaliGemma2 — 검출

미학습 이미지 10장, `per_class` 추론:

![PaliGemma2 검출 결과 그리드](docs/images/paligemma2/det_samples.jpg)

> 리소스 제약으로 **가능성만 확인 후 학습 중간에 멈춘** 시점의 결과입니다.

**학습 곡선** — negative 샘플 포함 파인튜닝:

![PaliGemma2 검출 학습 loss](docs/images/paligemma2/det_loss.jpg)

### 🟣 Qwen3-VL — 패션 속성 인식

**예측 결과 (GT vs Prediction)**

![Qwen3-VL 패션 속성 정답 샘플](docs/images/qwen3vl/fashion_attr_samples.jpg)

**학습 곡선**

![Qwen3-VL 패션 속성 학습 loss](docs/images/qwen3vl/fashion_attr_loss.png)

### 🟢 nanoVLM — FineVision 스크래치 학습

학습분포 held-out(dvqa 미학습 tail), greedy:

![nanoVLM in-distribution 정답 샘플](docs/images/nanovlm/infer_indist_dvqa_ckpt12000.jpg)

> 18개 FineVision subset 을 백본(SigLIP2+SmolLM2)에서 스크래치로 12,000스텝 학습했습니다
> (멀티턴 패킹 + 과적합 완화 수정 적용). 학습분포에선 near-perfect(dvqa held-out 10/10),
> 상식추론(aokvqa OOD)은 학습 믹스에 없어 약합니다 — 진단은 [LAB/nanovlm.md](LAB/nanovlm.md) 10절.

**학습 곡선** — train loss / token-acc (12,000스텝 완주):

![nanoVLM 학습 loss](docs/images/nanovlm/loss.jpg)

### 🔵 SmolVLM2 — FineVision 파인튜닝

자유형 캡션/VQA, greedy:

![SmolVLM2 추론 샘플](docs/images/smolvlm2/infer_samples.jpg)

> 공식 `SmolVLM2-2.2B-Instruct` 가중치에서 FineVision 파인튜닝(12,000스텝). 옷·색·자세·소품까지
> 세밀하고 일관된 묘사를 냅니다. ("from scratch" 는 **모델 코드 재구현**을 뜻하며, 가중치는 공식
> base 에서 시작한 파인튜닝이라 nanoVLM 의 밑바닥 학습과 성격이 다릅니다.)

**학습 곡선** — train / eval loss (12,000스텝 완주):

![SmolVLM2 학습 loss](docs/images/smolvlm2/loss.jpg)

### 📊 통합 평가 — MMStar (ours vs 공식 base)

두 재구현 모델을 **동일 harness**(likelihood, n=1500)로 공식 원본과 비교했습니다.
이 저장소의 [eval/eval_nanovlm_bench.py](eval/eval_nanovlm_bench.py) ·
[eval/eval_smolvlm2_bench.py](eval/eval_smolvlm2_bench.py) 로 그대로 재현됩니다.

![MMStar 4모델 비교](docs/images/compare_mmstar.jpg)

| 모델 | 초기화 | MMStar |
|---|---|---|
| **SmolVLM2 2.2B (ours)** | 공식 base + FineVision FT | **43.1%** |
| SmolVLM2 2.2B (base) | 공식 원본 `SmolVLM2-2.2B-Instruct` | 42.4% |
| nanoVLM 450M (base) | 공식 원본 `lusxvr/nanoVLM` | 37.5% |
| **nanoVLM 450M (ours)** | 백본 + 랜덤커넥터 → FineVision | 34.2% |
| random (4지선다) | — | 25% |

- **SmolVLM2**: 공식 base 파인튜닝이라 **ours(43.1) ≥ base(42.4)** — 우리 재구현 코드로 base 를 돌려도 42.4% 라 코드 충실성도 뒷받침됩니다.
- **nanoVLM**: **ours(34.2) < base(37.5)** — 격차의 원인은 **데이터 차이**입니다(base=FineVision 전체 24M·200+소스 vs ours=18소스 ~1M). 자세히 [LAB/nanovlm.md](LAB/nanovlm.md) 10.17.

---

## 📦 설치

```bash
# 1) conda 환경 생성 · 활성화
conda create -n pierrot-infer python=3.10 -y
conda activate pierrot-infer

# 2) 의존성 설치
cd Pierrot-VLM
pip install -r requirements.txt
```

---

## 🔮 추론

`--model` 에는 **HF Hub id** 와 **로컬 학습 산출물(final/)** 을 모두 넣을 수 있습니다.
미지정 시 [infer/defaults.py](infer/defaults.py) 의 기본 체크포인트를 씁니다.

```bash
# ── PaliGemma2 ──
# 공개 사전학습 모델로 즉시 추론
python infer/infer_paligemma2.py --model google/paligemma2-3b-pt-224 \
    --images cat.jpg --prompt "caption en"
# 검출: loc 토큰을 박스로 파싱 + 시각화 저장(<원본이름>_pred.jpg)
python infer/infer_paligemma2.py --model ./outputs/paligemma2_ft/final --images img.jpg \
    --prompt "detect earrings ; necklaces ; watch" --detect --save-dir preds

# ── nanoVLM ──
python infer/infer_nanovlm.py --model lusxvr/nanoVLM-450M --image img.jpg --prompt "What is this?"

# ── SmolVLM2 ──
python infer/infer_smolvlm2.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --image img.jpg --prompt "Describe this image."

# ── Qwen3-VL ──
python infer/infer_qwen3vl.py --model Qwen/Qwen3-VL-2B-Instruct --image img.jpg --prompt "What is this?"
# 검출: 텍스트 좌표를 박스로 파싱 + 시각화 저장
python infer/infer_qwen3vl.py --model ./outputs/qwen3vl_ft/final --image img.jpg \
    --prompt "detect earrings ; necklaces ; watch" --detect --save-viz out.jpg
# 패션 속성(부위별 JSON): roi 를 픽셀 좌표로 복원해 출력
python infer/infer_qwen3vl.py --model ./outputs/qwen3vl_fashion/final --image img.jpg --fashion

# ── Qwen3.5 ──
python infer/infer_qwen35.py --model Qwen/Qwen3.5-4B --image img.jpg --prompt "What is this?"
```

**모델별 주요 인자**

| 인자 | 해당 모델 | 설명 |
|---|---|---|
| `--detect` / `--save-viz` | paligemma2 · smolvlm2 · qwen3vl | 좌표 출력을 박스로 파싱하고 그려 저장 |
| `--fashion` | qwen3vl | 패션 속성 JSON 파싱 + roi(0~999) → 픽셀 복원 |
| `--max-pixels` / `--min-pixels` | qwen3vl · qwen35 | 이미지 토큰 예산(≈ `max_pixels`/1024 토큰) |
| `--max-splits-per-side` | smolvlm2 | 한 변 최대 타일 수(작은 물체 ↔ 시퀀스 길이 트레이드오프) |
| `--dtype` | 전체 | `bfloat16`(기본) / `float16` / `float32` |

> ⚠️ **전처리 값은 학습과 같아야 합니다.** 이미지 토큰 수를 정하는 손잡이(qwen 계열
> `max_pixels`, SmolVLM2 타일 분할)가 달라지면 시퀀스 구성이 달라져 품질이 떨어집니다.
> 학습 산출물에는 이 값이 sidecar(`<모델>_preprocessor.json`)로 동봉되어 **자동 상속**되므로,
> CLI 로 굳이 지정하지 않는 편이 안전합니다.

### 파이썬에서 직접

```python
from PIL import Image
from pierrot.models.qwen3vl import load_pretrained

model, processor = load_pretrained("Qwen/Qwen3-VL-2B-Instruct", device="cuda")
model.eval()

enc = processor(images=[Image.open("img.jpg").convert("RGB")], text=["Describe this image."])
enc = {k: v.to("cuda") for k, v in enc.items()}
out = model.generate(enc["input_ids"], enc["pixel_values"], enc["image_grid_thw"],
                     enc["attention_mask"], max_new_tokens=200,
                     eos_token_id=processor.eos_token_id)
print(processor.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## 📈 평가

MMStar 객관식 정확도를 **자체 harness**(lmms-eval 불필요)로 잽니다. 기본은 `likelihood`
모드 — 각 선택지 글자의 로그확률을 비교하므로, 모델이 장황하게 답해도 공정하게 채점됩니다.

```bash
# nanoVLM (기본: MMStar val 1500문항 전량)
python eval/eval_nanovlm_bench.py --ckpt lusxvr/nanoVLM-450M
python eval/eval_nanovlm_bench.py --ckpt ./outputs/nanovlm_mix/final --limit 200   # 스모크

# SmolVLM2
python eval/eval_smolvlm2_bench.py --ckpt HuggingFaceTB/SmolVLM2-2.2B-Instruct

# SmolVLM2 자유형 추론 → 큰 폰트 단일 HTML 뷰어(이미지 임베드)
python eval/infer_smolvlm2_viewer.py --model ./outputs/smolvlm2_finevision/final \
    --glob 'images/*.jpg' --limit 6
```

**공식 구현과의 수치 대조**(opt-in, 공개 가중치 다운로드 필요):

```bash
python tools/parity_qwen3vl.py     # transformers>=4.57 필요
python tools/parity_qwen35.py      # transformers>=5.1  필요
```

전처리(input_ids·grid·pixel_values) → 비전 타워 출력 → 전체 로짓 argmax → greedy 생성까지
네 단계를 순서대로 대조하고, 하나라도 어긋나면 실패로 종료합니다.

---

## 📁 디렉토리 구조

```
Pierrot-VLM/                    # git clone 으로 받는 디렉토리 이름
├── infer/                      # ★ 모델별 추론 엔트리포인트
│   ├── defaults.py             #   추론 기본값 단일 소스(체크포인트·dtype·전처리 예산)
│   ├── infer_paligemma2.py     #   검출(loc 토큰) 파싱 · GT 비교 · 스텝별 로그
│   ├── infer_nanovlm.py
│   ├── infer_smolvlm2.py
│   ├── infer_qwen3vl.py        #   검출 + 패션 속성(JSON·roi 복원)
│   └── infer_qwen35.py
├── eval/                       # 벤치마크 · 뷰어
│   ├── eval_nanovlm_bench.py   #   MMStar 정확도(likelihood/generate)
│   ├── eval_smolvlm2_bench.py  #   〃 (SmolVLM2 API)
│   └── infer_smolvlm2_viewer.py#   자유형 추론 → 단일 HTML 뷰어
├── tools/
│   ├── parity_qwen3vl.py       # 공식 HF 구현 대비 수치 대조(opt-in)
│   └── parity_qwen35.py
├── requirements.txt
├── LAB/                        # 실험 노트 5편 (아래 참조)
├── docs/images/                # LAB · README 그림
└── pierrot/
    └── models/                 # 알고리즘 패키지 (추론 경로만)
        ├── paligemma2/         #   SigLIP-So400m + 선형 프로젝터 + Gemma2
        │   ├── config.py       #     SigLIP/Gemma2/PaliGemma2 설정 (HF config.json 과 1:1)
        │   ├── modeling/       #     siglip · gemma2 · projector · paligemma2(prefix-LM + generate)
        │   ├── processor.py    #     이미지 전처리 + <image> placeholder 프롬프트
        │   ├── detection.py    #     loc 토큰 파싱 & 시각화
        │   └── weights.py      #     HF Hub 다운로드 + safetensors 로드
        ├── nanovlm/            #   SigLIP2 ViT + 픽셀셔플 + SmolLM2
        │   ├── transforms.py   #     동적 리사이즈 + 타일 분할(+global 타일)
        │   └── ...
        ├── smolvlm2/           #   SigLIP + 픽셀셔플 커넥터 + SmolLM2 (Idefics3식 타일링)
        ├── qwen3vl/            #   동적해상도 ViT + DeepStack + 패치머저 + Qwen3(M-RoPE)
        └── qwen35/             #   동적해상도 ViT + Gated DeltaNet 3:1 하이브리드 디코더
```

학습 배포본과 달리 `args/` · `training/` · `pierrot/core`(Accelerate 엔진) · `pierrot/data` ·
모델별 `dataset.py` · `spec.py` 가 없습니다. 손실 계산·활성화 체크포인팅·파라미터 그룹 같은
학습 경로도 모델 코드에서 빠져 있습니다 — **추론 수치는 학습 저장소와 동일**합니다.

---

## 🧪 실험 노트 (LAB)

모델별 설계·코드 흐름·실험 기록입니다. 성공뿐 아니라 **함정과 실패**도 함께 적었습니다.

| 문서 | 내용 |
|---|---|
| [LAB/paligemma2.md](LAB/paligemma2.md) | prefix-LM 마스크 · 선형 프로젝터 · **검출 학습 실험**(negative 샘플로 오탐 해결) |
| [LAB/nanovlm.md](LAB/nanovlm.md) | 픽셀셔플 · 타일링 · **이식 5-way 리뷰 · 멀티턴 패킹 수정 · 과적합 · 평가셋 진단** |
| [LAB/smolvlm2.md](LAB/smolvlm2.md) | Idefics3 타일링 · **6-way 리뷰 · pad/config/RoPE 함정 · MMStar 평가** |
| [LAB/qwen3vl.md](LAB/qwen3vl.md) | 동적해상도 · DeepStack · M-RoPE · **패션 속성 학습 실험** |
| [LAB/qwen35.md](LAB/qwen35.md) | Gated DeltaNet 하이브리드 디코더 스크래치 · **공식 체크포인트 텐서 1:1 검증** |

문서에서 `args/` · `training/` · `tools/build_*` 를 가리키는 링크는 **학습 저장소**
[Pierrot-VLM-Lab](https://github.com/Pierrot-vision/Pierrot-VLM-Lab) 으로 연결됩니다 — 이 배포본에는
없는 파일들입니다.

---

## 📚 참조 (Reference)

관련 논문 정리 · 리뷰 → [Pierrot-vision/Reading-Papers — VLM](https://github.com/Pierrot-vision/Reading-Papers#-vlm)

- **[Pierrot-VLM-Lab](https://github.com/Pierrot-vision/Pierrot-VLM-Lab)** — 학습 저장소(전체 코드)
- **[Pierrot-VLM-OCR](https://github.com/Pierrot-vision/Pierrot-VLM-OCR)** — 문서 파싱(OCR) 추론 배포본

---

## 📄 라이선스

이 프로젝트(코드 · 문서)는 **CC BY-NC-SA 4.0** 을 따릅니다 — 출처 표기 · 비영리 · 동일조건 변경허락. 자세한 내용은 [LICENSE](LICENSE) 참고. (사용된 서드파티 데이터셋 · 모델 · 라이브러리는 각자의 라이선스를 따릅니다.)

> ⛔ **상업적 사용 금지 (NonCommercial)** — 비영리 목적으로만 사용할 수 있습니다. 상업적 이용이 필요하면 별도 문의 바랍니다.

---

## 📮 문의

- [메일](mailto:peternara@naver.com) 또는 [GitHub Issue](https://github.com/Pierrot-vision/Pierrot-VLM/issues) 를 통해 관련 질문·문의 부탁드립니다. 대답할수 있는 내용이라면 성실이 답변드리겠습니다.
- 참고로, 이미 GitHub(README · 코드 · 문서)에 있는 내용을 다시 문의하시면 답을 드리지 못할 수 있는 점 양해 부탁드립니다.
