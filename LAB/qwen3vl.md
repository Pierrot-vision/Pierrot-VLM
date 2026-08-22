# Qwen3-VL — 코드 기반 전체 플로우

Pierrot-VLM-Lab 의 Qwen3-VL 구현을 **실제 코드 흐름** 그대로 따라가는 문서다.
학습 1스텝과 추론 1회가 코드에서 어떤 함수를 어떤 순서로 통과하는지 추적한다.

> 이 저장소는 **추론 배포본**이다. 학습 스크립트(`training/`)·하이퍼파라미터(`args/`)·
> 데이터 빌더는 들어 있지 않으므로, 본문에서 그런 파일을 가리키는 링크는 학습 저장소
> [Pierrot-VLM-Lab](https://github.com/Pierrot-vision/Pierrot-VLM-Lab) 로 연결된다.

> 백본 요약: **동적 해상도 ViT(+DeepStack) → 패치 머저 → Qwen3(M-RoPE·QK-Norm) 언어 디코더**.
> 세 선배 모델(PaliGemma2/nanoVLM/SmolVLM2)과 가장 크게 다른 점은 **타일 분할이 없다**는 것.
> 이미지를 정사각 타일로 자르지 않고 **원본 종횡비를 유지한 채 32 배수로 리사이즈**해
> 가변 격자(grid_thw)로 보며, 그래서 **이미지 토큰 수가 이미지마다 다르다**. prefix-LM
> 마스크는 없고 ChatML(`<|im_start|>`) 기반 **일반 causal LM**, 라벨 마스킹(-100)으로
> assistant(정답) 토큰에만 손실을 건다.

---

## 0. 파일 지도

```
Pierrot_VLM/
├── args/qwen3vl.py                 # 하이퍼파라미터 단일 소스 (PARAMS dict → 평탄한 args)
├── training/train_qwen3vl.py       # 학습 진입점
├── infer/infer_qwen3vl.py          # 추론 진입점
└── pierrot/
    ├── core/                          # 모델-비의존 학습 인프라 (다른 모델과 공유)
    │   ├── registry.py                #   ModelSpec 인터페이스 + 레지스트리
    │   ├── engine.py                  #   Trainer (Accelerate 학습 루프)
    │   └── scheduler.py               #   warmup + cosine LR
    ├── data/                          # 공용 데이터 소스 (모델 간 공유)
    │   ├── jsonl.py                   #   JsonlDataset ({image,prefix,suffix})
    │   ├── coco.py                    #   CocoDetectionSource
    │   └── detection.py               #   DetectionPromptDataset (프롬프트 모드 인덱싱)
    └── models/qwen3vl/
        ├── config.py                  # Qwen3VLVisionConfig / Qwen3VLTextConfig / Qwen3VLConfig
        ├── modeling/
        │   ├── vision.py              #   동적 해상도 ViT + bilinear 위치보간 + 패치머저 + DeepStack
        │   ├── text.py                #   Qwen3 디코더 (RMSNorm+M-RoPE+QK-Norm+GQA+SwiGLU, +KVCache)
        │   └── qwen3vl.py             #   최상위: 병합(masked_scatter) + M-RoPE 위치 + causal forward + loss + generate
        ├── processor.py               # 동적 해상도 패치화(smart_resize) + ChatML 토크나이즈 + 라벨 생성
        ├── dataset.py                 # JSONL/COCO 어댑터 + collate(패치 패킹)
        ├── detection.py               # 텍스트 좌표 파싱/시각화
        ├── weights.py                 # HF Hub 다운로드 + safetensors 로드 (raw config, AutoConfig 불필요)
        └── spec.py                    # 레지스트리 어댑터 (@register_model "qwen3vl")
```

두 종류의 "config" 를 구분하는 것이 이해의 핵심이다:

| 종류 | 클래스 | 어디서 오나 | 무엇 |
|---|---|---|---|
| **실험 설정** | `args` (평탄 네임스페이스) | [args/qwen3vl.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/args/qwen3vl.py) | lr, batch, epochs, 어느 체크포인트, 동결, **max_pixels** |
| **모델 구조** | `Qwen3VLConfig` 등 | 체크포인트 `config.json` (raw) | hidden_size, 레이어 수, head 수, M-RoPE 배분, DeepStack 인덱스... (자동) |

> **다른 세 모델과 완전히 분리**되어 있다. 파일 이름이 전부 `_qwen3vl` 접미사를 달고,
> 레지스트리 이름 `'qwen3vl'` 로만 연결된다. SmolVLM2 와 달리 config 복원에 HF AutoConfig 가
> 필요 없다 — Qwen3-VL 의 config.json 은 모든 구조 키를 명시하므로 raw JSON 파싱만으로 충분하다
> (덕분에 Qwen3-VL 을 아직 모르는 transformers<4.57 환경에서도 이 구현이 그대로 동작한다).

---

## 1. 학습 전체 플로우 (training/train_qwen3vl.py 진입 → 저장)

```
python training/train_qwen3vl.py
   │
   ├─(a) from args.qwen3vl import args      # PARAMS → 평탄한 args (단일 소스)
   ├─(b) _preflight(args)                   # 다운로드 前 데이터/이미지 경로 검사 + 픽셀 예산 출력
   ├─(c) spec = get_model_spec('qwen3vl')   # 레지스트리에서 Qwen3VLSpec 선택
   ├─(d) model, processor = spec.build(args)     # 아래 2절
   ├─(e) train_ds = spec.build_dataset(args,'train',processor)
   │     collate  = spec.collate_fn(processor,args)
   ├─(f) DataLoader(train_ds, collate_fn=collate, ...)
   └─(g) Trainer(model, loader, args, processor=...).fit()   # Accelerate 루프(다른 모델과 공유)
             └ 저장: output_dir/final/{model.pt, config.json, tokenizer, qwen3vl_preprocessor.json}
```

---

## 2. 모델 생성 — `spec.build(args)`

[spec.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/pierrot/models/qwen3vl/spec.py) 의 흐름 (파인튜닝 = `pretrained` 지정 기준):

```
build(args)
   ├─ proc_kwargs = {min_pixels, max_pixels, system_prompt}    # None 이면 sidecar/공식값 상속
   ├─ pretrained 있음 → weights.load_pretrained(...)           # 아래
   │     ├ resolve_model_dir       # 로컬 or HF Hub snapshot_download
   │     ├ config_from_json        # config.json(raw) → Qwen3VLConfig, rope_scaling→M-RoPE 승격
   │     ├ build_processor         # 토크나이저 + 픽셀 예산(우선순위: CLI>sidecar>fallback>공식)
   │     ├ config.image_token_id = processor.image_token_id
   │     ├ Qwen3VLForConditionalGeneration(config)
   │     ├ *.safetensors 합쳐 load_state_dict(strict=False)
   │     ├ tie_word_embeddings=True → tie_weights()   # 2B 는 lm_head 가 체크포인트에 없음
   │     └ _verify_load            # 누락/여분 키 엄격 검사(조용한 랜덤 초기화 방지)
   ├─ _apply_freezing(model,args)  # freeze_vision(블록/patch/pos) · freeze_projector(머저)
   └─ gradient_checkpointing_enable()  # 비전 블록 + 언어 디코더
```

`pretrained=None`(스크래치)이면 `model_extra` 의 text/vision config 로 랜덤 초기화하고,
프로세서를 먼저 만들어 `<|image_pad|>` id 를 config 에 반영한다.

---

## 3. 데이터 → 배치 플로우

### 3.1 데이터 형식 (JSONL)

```json
{"image": "images/cat.jpg", "prefix": "질문/지시문", "suffix": "정답"}
```

### 3.2 Dataset ([dataset.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/pierrot/models/qwen3vl/dataset.py))

- **jsonl**: 공용 `JsonlDataset` 재사용 → `{image(PIL), prefix, suffix}`
- **coco**: `Qwen3VLDetectionDataset`(공용 `DetectionPromptDataset` 상속). suffix 를
  `"{class}: x0,y0,x1,y1 ; ..."`(0~999 정규 정수 좌표)로 표기. ★ 좌표가 **원본 크기 기준**이라
  이미지마다 다른 동적 리사이즈 격자와 무관하게 라벨이 일관된다.

### 3.3 동적 해상도 패치화 + ChatML + 라벨 ([processor.py](../pierrot/models/qwen3vl/processor.py))

`encode_one(image, prefix, suffix)` — 한 샘플:

```
① process_image(image)
     smart_resize(h,w, factor=patch×merge=32, min_pixels, max_pixels)   # 종횡비 유지, 32 배수
       → BICUBIC resize → [-1,1] 정규화
       → 머저 블록(m×m) 우선 순서로 패치를 펼침 → (n_patches, patch_dim=C·T·p·p)
       → 격자 grid=(1, h/p, w/p)
② 이미지 문자열 = <|vision_start|> + <|image_pad|>×(t·h·w/m²) + <|vision_end|>   # placeholder 개수=이미지 토큰 수
③ ChatML 조립:
     <|im_start|>user\n{이미지문자열}{prefix}<|im_end|>\n<|im_start|>assistant\n{suffix}<|im_end|>
④ 라벨: user/이미지 구간 = -100, assistant(정답)+끝 <|im_end|> 만 정답 id (suffix-only 손실)
```

> **★ max_pixels 가 이 모델의 유일하고 가장 강력한 손잡이다.** 이미지 토큰 수 ≈ max_pixels/1024
> (1024 = (patch16×merge2)²). 공식 기본값(16,777,216)은 학습엔 과해서, args 기본은 327,680
> (=320×32×32, 이미지당 최대 320 토큰)으로 둔다. OOM 이면 낮추고, 작은 물체 검출이면 올린다. 이 값은
> 추론에서도 같아야 하므로 학습 산출물에 `qwen3vl_preprocessor.json`(sidecar)으로 동봉된다.

### 3.4 collate — 패치 패킹 ([processor.py](../pierrot/models/qwen3vl/processor.py) `collate_encoded`)

```
input_ids/attention_mask/labels : (B, Lmax) 우측 패딩
pixel_values                    : (총패치수, patch_dim)  ← 패딩 없이 이어붙임(이미지마다 크기가 달라 패딩 불가·불필요)
image_grid_thw                  : (이미지수, 3)          ← 이미지 경계 정보
```

> 다른 모델은 타일을 `(B, max_tiles, ...)` 로 zero 패딩하지만, Qwen3-VL 은 크기가 제각각이라
> 패킹한다. 배치 순서대로 이어붙이고, 모델의 `masked_scatter` 가 같은 순서로 소비한다.

---

## 4. 모델 forward — `Qwen3VLForConditionalGeneration.forward`

### 4.1 준비 (`_prepare_inputs`)

```
inputs_embeds = embed_tokens(input_ids)                         # (B, T, D)
image_embeds, deepstack = visual(pixel_values, image_grid_thw)  # 4.2
inputs_embeds, visual_masks = _merge_image_features(...)        # <|image_pad|> 자리에 masked_scatter, 길이 보존
position_ids, next_pos = get_rope_index(...)                    # 4.4 (3, B, T) M-RoPE 위치
```

### 4.2 동적 해상도 비전 타워 ([vision.py](../pierrot/models/qwen3vl/modeling/vision.py))

```
patch_embed (Conv3d)                    (S, patch_dim) → (S, D)
+ 위치 임베딩: 48×48 학습 테이블을 격자 (h,w) 로 bilinear 4-tap 보간(합=1)해 덧셈   ★ NaFlex 방식
2D RoPE(cos/sin) 준비                    머저 블록 순서와 일치하는 (h,w) 인덱스
블록 × depth:
   이미지 경계(seq_lengths)마다 끊어 SDPA  ← 서로 다른 이미지 패치가 섞이지 않음(마스크 없이 Flash 가능)
   deepstack_visual_indexes 층 → 별도 머저로 중간 특징 수집   ★ DeepStack
merger: m×m 이웃 패치를 reshape 로 접어 (S/m², D·m²) → 선형 → (S/m², Dout=언어 hidden)
```

- **본 머저**는 합치기 **전**(D)에서 LayerNorm, **DeepStack 머저**는 합친 **후**(D·m²)에서
  LayerNorm(`use_postshuffle_norm`). norm 파라미터 크기가 달라 공개 가중치와 정확히 맞춰야 한다.

<a id="projector"></a>
#### 4.2.1 패치 머저 (Patch merger)

비전 타워 마지막 단계다: **m×m 이웃 패치를 reshape 로 접어**(`(S, D) → (S/m², D·m²)`) 선형
투영으로 언어 hidden 차원에 맞춘다 — 이미지 토큰을 **m² 배 압축**한다(예: m=2 → 1/4).
`deepstack_visual_indexes` 층의 중간 특징은 **별도 DeepStack 머저**로 모아 병합 단계에서 더한다.

### 4.3 이미지 병합 (`_merge_image_features`)

`masked_scatter` 로 `<|image_pad|>` 위치를 이미지 임베딩으로 교체(길이 보존). placeholder
개수 ≠ 이미지 임베딩 개수면 **즉시 ValueError**(프로세서 grid 와 모델 merge_size 불일치 감지).

### 4.4 position_ids — M-RoPE (`get_rope_index`)

위치가 스칼라가 아니라 **(시간 t, 높이 h, 너비 w) 3축**이다:

```
텍스트 L 토큰  : 세 축 모두 pos, pos+1, ..., pos+L-1     → pos += L
이미지 격자    : t/h/w 좌표를 meshgrid 로 펼쳐 pos 더함  → pos += max(h,w)/m   (차지한 위치 폭)
패딩 위치      : 계산 제외, 0 으로 둠(어텐션 마스크가 가림)
```

`next_pos`(각 샘플의 다음 위치)는 생성 단계에서 이어 쓸 시작 위치로 반환된다.

### 4.5 Qwen3 언어 디코더 ([text.py](../pierrot/models/qwen3vl/modeling/text.py))

Llama 계열과 다른 두 가지 + DeepStack:
- **QK-Norm** — q/k 를 head_dim 에서 RMSNorm 한 **뒤** RoPE 적용(Qwen3 고유).
- **M-RoPE** — head_dim/2 주파수를 mrope_section=[24,20,20] 로 T/H/W 에 배분, **interleaved**
  ([THWTHW...]) 배치. `rotary_emb._combine_axes` 가 축을 하나의 각도 벡터로 합친다.
- **DeepStack 주입** — 앞쪽 N(=DeepStack 층 수)개 디코더 레이어 출력의 이미지 토큰 위치에
  비전 중간층 특징을 더한다(`_deepstack_add`).

### 4.6 손실 (suffix-only, 메모리 절약형)

labels 를 1칸 시프트해 `-100` 이 아닌 위치(=정답)의 hidden 만 `lm_head` 에 통과 → CE.
전체 `(B,T,V)` 로짓을 만들지 않아 vocab 15만짜리 Qwen 에서 메모리를 크게 아낀다.

---

## 5. 학습 루프 — `Trainer.fit` (Accelerate)

다른 세 모델과 **완전히 동일한 엔진**([engine.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/pierrot/core/engine.py))을 쓴다. 모델은
`forward(**batch) → {"loss"}` 규약만 지키면 되므로 Qwen3-VL 도 그대로 학습된다. grad accum /
bf16 / DDP·FSDP / 체크포인트 재개 / cosine LR / NaN 가드 모두 공유. 저장 시 프로세서의
`save_preprocessor_config` 훅으로 `qwen3vl_preprocessor.json` 을 동봉한다.

---

## 6. 추론 전체 플로우 (infer/infer_qwen3vl.py)

```
python infer/infer_qwen3vl.py --model Qwen/Qwen3-VL-2B-Instruct --image cat.jpg --prompt "Describe this image."
   │
   ├─ load_pretrained(model, min_pixels/max_pixels=None → sidecar/args/공식 상속)   # ★
   ├─ inputs = processor([image], text=[prompt])    # suffix 없음 → labels 없음
   ├─ model.generate(...)                           # KV 캐시 autoregressive (아래)
   └─ 프롬프트 뒤 새 토큰만 tokenizer.decode → 출력 (+ --detect 시 박스 파싱/시각화)
```

> **★ 전처리 일치**: `max_pixels` 를 학습과 똑같이 맞춰야 이미지 토큰 수가 같아진다. 안 맞추면
> 파인튜닝 때와 달라져 성능이 떨어진다. CLI 미지정(None)이면 모델 sidecar → args → 공식값 순으로
> 상속한다(Hub repo id 로 올린 파인튜닝 모델도 sidecar 가 이긴다).

`generate` ([qwen3vl.py](../pierrot/models/qwen3vl/modeling/qwen3vl.py), 배치=1 · 패딩 없는 프롬프트 가정):
```
프리필(step 0):
   이미지 병합·DeepStack 주입된 프롬프트 → 언어 디코더 한 번 → KVCache + 마지막 logits → 첫 토큰
   next_pos(M-RoPE 다음 위치) 확보
디코드(step ≥ 1):
   새 토큰 1개만 임베딩, 위치=next_pos+step (세 축 동일), kv_cache 재사용 → 다음 토큰
   (새 토큰은 이미지가 아니므로 DeepStack 주입 없음)
   greedy(argmax) 또는 top-p 샘플링, <|im_end|>/<|endoftext|>(eos) 만나면 종료
```

---

## 7. 텐서 shape 흐름 요약 (Qwen3-VL-2B 기준)

```
이미지  smart_resize → (S, patch_dim=1536)            # S=총 패치 수(이미지마다 다름)
  └ 비전 24블록(+DeepStack 5/11/17층)  → (S, 1024)
  └ 패치 머저(m=2)                     → (S/4, 2048)   ─┐
  └ DeepStack 머저 × 3                 → (S/4, 2048)    │ 병합(masked_scatter)
텍스트  input_ids (B, T)                                │ <|image_pad|> 자리 교체, 길이 보존
  └ embed          → (B, T, 2048)                     ─┘
                   → Qwen3 28층(M-RoPE·QK-Norm, 앞 3층 DeepStack 주입) → (B, T, 2048)
                   → (suffix 위치만) lm_head → logits_kept (N, 151936)
                   → CE(ignore=-100) → loss (스칼라)
```

| 이름 | 값 (Qwen3-VL-2B) |
|---|---|
| 패치 크기 / 시간패치 / 머저 | 16 / 2 / 2 (m²=4배 압축) |
| 비전 hidden / depth / heads | 1024 / 24 / 16 · DeepStack 층 [5,11,17] |
| 위치 임베딩 격자(보간 소스) | 48×48 (2304) → bilinear 4-tap |
| Qwen3 hidden / 레이어 | 2048 / 28 |
| Qwen3 헤드 (Q/KV) | 16 / 8 (GQA) · head_dim **128** · rope_theta 5e6 |
| M-RoPE 배분 [t,h,w] | [24,20,20] · interleaved · **QK-Norm** |
| 이미지 토큰 수 | **가변** ≈ max_pixels/1024 (args 기본 max_pixels → 최대 320) |

> 크기(2B/4B), Instruct/Thinking 변형은 전부 config.json 에서 온다 — args 의 `pretrained` 만 바꾸면 된다.

---

## 8. 한눈에 보는 전체 그림

```
                 args/qwen3vl.py (PARAMS 단일 소스)
                        │  args.lr, args.pretrained, max_pixels, ...
        ┌───────────────┴──────────────────┐
        ▼                                   ▼
   spec.build(args)                    Trainer(args)  ── Accelerate 루프
   ├ weights.load_pretrained            └ for batch: model(**batch).loss
   │   ├ config_from_json (raw, AutoConfig 불필요)  ├ backward / clip / step
   │   └ *.safetensors → strict=False               └ save_pretrained(final + sidecar)
   ├ Qwen3VLForConditionalGeneration                       │
   │   ├ visual (동적 ViT + DeepStack + 머저)              ▼
   │   └ language_model (Qwen3 M-RoPE·QK-Norm)   output_dir/final/{model.pt, config.json,
   └ Qwen3VLProcessor (동적 패치화 + ChatML)                tokenizer, qwen3vl_preprocessor.json}
        │                                                   │
        └────── DataLoader(collate=패치 패킹) ──────────────►  infer/infer_qwen3vl.py: load_pretrained → generate
                                                              (max_pixels 를 sidecar 로 학습과 일치)
```

**핵심 5가지** (선배 세 모델 대비 Qwen3-VL 구현의 특징):
1. **타일 분할 없는 동적 해상도** — smart_resize 로 종횡비 유지 32 배수, 이미지 토큰 수 가변 (3.3·4.2)
2. **bilinear 위치 임베딩 보간** — 48×48 고정 테이블을 임의 격자에 4-tap 보간 (4.2)
3. **DeepStack** — 비전 중간층 특징을 앞쪽 디코더 레이어에 재주입 (4.2·4.5)
4. **M-RoPE + QK-Norm** — 3축(시간/높이/너비) 위치 + Qwen3 특유의 헤드 정규화 (4.4·4.5)
5. **패치 패킹 배치** — 크기 제각각 이미지를 패딩 없이 이어붙이고 grid_thw 로 경계 관리 (3.4)

> **검증**: `tests/test_qwen3vl.py` 는 소형 랜덤 모델로 기계(보간·머저·DeepStack·M-RoPE·loss·generate·
> 저장/재로드)를 확인하고, 전처리는 공식 `Qwen2VLImageProcessor` 와 격자·픽셀 일치를 검사한다.
> 실제 2B 가중치 parity(공식 HF 구현 대비 전처리·비전 타워·로짓 argmax·greedy 생성 일치)는
> **opt-in 스크립트 [tools/parity_qwen3vl.py](../tools/parity_qwen3vl.py)** 로 재현한다
> (`transformers>=4.57` 필요 — 미지원 환경이면 skip). attention/M-RoPE/전처리를 수정하면 이걸 돌려
> 회귀를 잡는다. 최근 실행: 전 항목 PASS (머저/DeepStack max|diff|=0.0, argmax 100%, greedy 동일).

---

# 9. 패션 속성 학습 실험 노트

"코드가 어떻게 도는가"가 아니라 **"왜 이렇게 구성했는가"** 를 남긴다.
([paligemma2.md 9장](paligemma2.md#9-검출-학습-실험-노트)과 같은 형식.)

> **한 줄 요약** — 이미지 → 부위별 속성 JSON 을 통째로 생성하게 파인튜닝했다.
> loss 는 13에폭까지 미세 하강하지만 **정확도는 3에폭에서 수렴**(85.7%)했고,
> 그 과정에서 bf16 마스크 dtype 버그와 generate() 의 배치 버그 2개를 잡았다.

---

## 9.1 데이터셋

`<DATA_ROOT>/VLM`(자체 데이터셋 — 구성은 [Datasets.md](Datasets.md#3-attr) 참고) — 이미지·라벨 **792,120쌍** (소스A 398k · 소스B 257k · 소스C 137k).
라벨은 부위 그룹별 아이템 리스트이고, 각 아이템이 category·color·pattern·style·situation·
fit·sleeve_length·strap_style·length·neckline·season·material·feature 를 **동의어 리스트**로 갖는다.

실험 설계를 좌우한 성질 세 가지:

| 성질 | 실측 | 영향 |
|---|---|---|
| 이미지당 아이템 수 | 1개 94.3% · 2개 5.2% · 3개 0.4% | 출력이 짧다(중앙값 ~150토큰) → max_length 2048 여유 |
| **아이템 수 = 그룹 수** (전수 일치) | 한 그룹에 2개 아이템인 경우가 0 | **roi 없이도 아이템 식별 모호성 없음** → roi 제거 근거 |
| roi 분포 | 전체 이미지 박스 ~60%, 부분 박스 ~40% | 검출 확장 시 grounding 필요 → roi 보존판 별도 생성 |

변환은 [tools/build_fashion_jsonl.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/tools/build_fashion_jsonl.py) 하나로 한다 — 메타키(`origin_image_path`·`image_path`·
`size`·`hash` 등)를 [convert_one()](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/tools/build_fashion_jsonl.py#L54) 에서 제거하고, 그룹 리스트만
suffix 로 직렬화(compact JSON). [include_roi](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/tools/build_fashion_jsonl.py#L31) 스위치로 **두 벌**을 만든다:

```
data/fashion_noroi/   roi 키 제거 (속성만) ← 현재 학습 사용
data/fashion/         roi 를 0~999 정규 좌표로 보존 (qwen3vl 검출 규약과 동일) — 예비
```

분할은 **eval 10,000(최종 홀드아웃·학습 금지) / val 5,000(학습 중 검증) / train 777,120**.
파일명 정렬 + seed 42 셔플이라 **roi 유무와 무관하게 같은 이미지가 같은 split** 에 들어간다
(나중에 roi 판으로 갈아타도 eval 오염 없음).

---

## 9.2 설계 선택 4가지

### ① 출력 = 라벨 JSON "그대로" (roi 제외)

요구사항이 "라벨에서 메타 필드만 뺀 형식으로 추론"이라 출력 스키마를 새로 설계하지 않았다.
roi 는 9.1 의 "아이템 수 = 그룹 수" 실측을 근거로 뺐다 — 속성 귀속이 모호해지지 않고,
좌표 회귀라는 어려운 과제가 사라져 속성 학습이 쉬워진다.

### ② prefix 단일 소스

지시문은 [build_fashion_jsonl.py 의 PARAMS['prefix']](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/tools/build_fashion_jsonl.py#L33) 하나에만 두고,
추론([infer_qwen3vl.py `--fashion`](../infer/infer_qwen3vl.py#L51), [tools/infer_fashion_batch.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/tools/infer_fashion_batch.py))이
그걸 **import 해서** 쓴다. 학습·추론 프롬프트 불일치로 형식이 안 나오는 사고를 원천 차단.

### ③ `max_pixels = 320×32×32` (이미지당 최대 320토큰)

속성 인식은 OCR·검출처럼 고해상도가 필요 없다. 320토큰이면 VRAM·속도 대비 충분했고,
레퍼런스 레포들(512~1280토큰)보다 작게 잡은 **의도된 선택**. roi 판으로 확장 시 재검토.

### ④ 하이퍼파라미터 = 오피셜 레시피 정렬

2U1/Qwen-VL-Series-Finetune·논문들과 대조해 lr 1e-5 · cosine+warmup 3% · wd 0 ·
유효배치 256(=32×1×분산) 로 맞췄다([args/qwen3vl.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/args/qwen3vl.py)). 유일한 의도적 차이는
**vision lr 차등(2e-6) 미적용** (엔진 훅 [optimizer_param_groups — engine.py:83](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/pierrot/core/engine.py#L83) 는
있으나 qwen3vl 모델에 미구현 — 남은 숙제).

---

## 9.3 잡은 버그 ① — bf16 SDPA 마스크 dtype

분산 학습 첫 스텝에서 `invalid dtype for bias` 크래시. 패딩 마스크를 `.float()` 로 만들어
bf16 쿼리와 dtype 이 어긋난 것([text.py:175](../pierrot/models/qwen3vl/modeling/text.py#L175)) —
batch>1(패딩 발생) + 순수 bf16 조합에서만 드러나는 버그라
이전 검증(배치 1·fp32 parity)을 전부 통과해 숨어 있었다.

```python
# modeling/text.py:175 — 수정 전 → 후
additive = (1.0 - m...float()) * torch.finfo(q.dtype).min      # fp32 마스크 (크래시)
additive = (1.0 - m....to(q.dtype)) * torch.finfo(q.dtype).min # 쿼리와 동일 dtype
```

모델 내 다른 `.float()` 들(RMSNorm·RoPE·비전 어텐션)은 fp32 계산 후 원 dtype 복귀하는
의도된 정밀도 처리라 유지.

---

## 9.4 학습 곡선 — 절벽, 그리고 긴 평탄

```python
# args/qwen3vl.py 발췌
'pretrained' : 'Qwen/Qwen3-VL-2B-Instruct',   # 지시학습 완료 모델에서 출발
'batch_size' : 32, 'grad_accum' : 1,          # 유효배치 256 (에폭 = 3,035 스텝)
'epochs'     : 50,                            # 상한만 크게 — 실제로는 조기 중단
```

![Qwen3-VL 패션 속성 학습 loss](../docs/images/qwen3vl/fashion_attr_loss.png)

*x축 step. 위=loss(train raw/ma50 + eval 500스텝마다 val 5,000장), 아래=LR(warmup 4,552스텝 후 cosine).*

| step (에폭) | train (ma50) | eval | 비고 |
|---|---|---|---|
| 0 | 2.85 | — | 스크래치면 ~11 에서 시작할 것 — 2.85 자체가 사전학습의 힘 |
| ~1,500 (0.5ep) | 0.15 | 0.18 | **절벽 끝** — 고정 JSON 골격(키·괄호) 암기 완료 |
| 3,035 (1ep) | 0.098 | 0.103 | 1에폭 체크포인트 |
| 9,105 (3ep) | 0.060 | 0.062 | **정확도 기준 사실상 정점** |
| 39,000 (13ep·중단) | 0.051 | **0.0542** | 최근 2,000스텝 변화 ±0.0001 — 평탄 |

**읽는 법**:
- 절벽은 성능이 아니라 **출력 엔트로피 구조** 때문이다. suffix 토큰의 대부분이 매번 똑같은
  JSON 보일러플레이트라 몇백 스텝에 외워지고, 남는 loss(~0.05)가 진짜 속성 판별 난이도다.
  (PaliGemma2 검출이 1.4 에서 바닥친 것과 대비 — 좌표 토큰은 본질 엔트로피가 높다.)
- train/eval 이 끝까지 밀착 → 과적합 없음. 그런데도 학습을 끊은 이유는 다음 절.

---

## 9.5 loss 는 내려가는데 정확도는 멈췄다

500스텝마다의 eval loss 만 보면 13에폭에도 "아직 내려가는 중"이다(0.0566→0.0542).
그러나 **홀드아웃 100장 필드 일치율**(동의어 교집합 판정)은 다르게 말한다:

| 체크포인트 | 필드 일치율 | 직전 대비 필드 개선/악화 | JSON 파싱 |
|---|---|---|---|
| step 3,035 (1ep) | 81.8% | — | 99/100 (반복 루프 1건) |
| step 6,070 (2ep) | 85.2% | **+54 / −23** | 100/100 |
| step 9,105 (3ep) | 85.7% | +12 / −7 | 100/100 |
| step 36,420 (12ep) | 85.4% | +13 / **−16 (순증 음수)** | 100/100 |

- 1→2에폭: eval loss 0.026 하락에 정확도 **+3.4%p** — loss 가 낮은 구간에선 작은 하락도
  "동률 근처 케이스가 뒤집히는" 효과라 체감이 크다.
- 3→12에폭: eval loss 0.008 하락에 정확도 **−0.3%p(노이즈 범위)** — 남은 loss 개선분이
  정답 확률의 마진만 키우고 argmax 는 바꾸지 못한다. PaliGemma2 의 "loss 는 내려가는데
  검출 수는 정체"([9.5](paligemma2.md#95-현재-학습-negative-포함))와 같은 패턴.
- **결론: 3에폭이 정점. 이후 9에폭(~20 GPU·h)은 낭비였다** — 가능성 확인이 목적이므로 중단.

오답의 성격: color 는 100/100, 오답은 인접 클래스(Sandals↔Slippers, Tank top↔Sleeveless
T-shirt)와 가림 속성(strap_style·fit)에 집중 — 라벨 노이즈도 발견(shoes 의 type 에 "Sling bag").

---

## 9.6 잡은 버그 ② — generate() 배치 추론

eval 1만 장 평가를 단건 루프로 돌리면 6~11시간이라 배치 추론이 필요했다.
기존 [generate() — qwen3vl.py:251](../pierrot/models/qwen3vl/modeling/qwen3vl.py#L251) 는
"배치=1, 패딩 없음" 가정이라 두 군데를 고쳤다:

| 버그 | 증상 | 수정 |
|---|---|---|
| 프리필 `hidden[:, -1]` | 오른쪽 패딩 시 짧은 샘플은 **pad 위치의 로짓**으로 첫 토큰 생성 | `attention_mask.sum(-1)-1` 로 샘플별 마지막 실토큰 gather ([qwen3vl.py:288](../pierrot/models/qwen3vl/modeling/qwen3vl.py#L288)) |
| EOS 종료 조건 | 전 샘플이 **같은 스텝에** EOS 여야 정지 | 샘플별 finished 마스크 — 끝난 샘플은 EOS 강제, 전부 끝나면 break ([qwen3vl.py:294](../pierrot/models/qwen3vl/modeling/qwen3vl.py#L294)) |

검증은 **배치 vs 단건 greedy 완전 일치**로 했다(같은 샘플 6개 문자 단위 대조).
단, bf16 배치 행렬곱은 누적 순서가 달라 로짓 끝자리가 흔들리고, **동률 근처 토큰이 드물게
뒤집힌다**(6건 중 0~1건) — HF 커뮤니티에도 보고된 현상으로 품질 영향은 없다.

**속도 실측** (GPU 1장, 학습과 공유 상태):

| 방식 | 1회 호출 | 장당 |
|---|---|---|
| 배치 1 | 3.4s | 3.4s |
| 배치 20 | ~11s | **0.3~0.6s (6~12배)** |

학습에선 배치 12→32 가 처리량 동일(compute 포화)했지만 **자기회귀 디코드는 memory-bound**
라 배치가 거의 공짜다 — 같은 가중치 읽기 한 번으로 20샘플의 토큰을 뽑는다.

---

## 9.7 운영 도구 — 실행마다 직전과 비교

에폭이 쌓일 때마다 "좋아졌나?"를 물어야 해서, 추론 도구에 비교를 내장했다:

```
tools/infer_fashion_batch.py     # ① outputs 의 최신 checkpoint-N 자동 선택
                                 # ② 홀드아웃 100장 배치 추론 (항상 같은 seed → 공정 비교)
                                 # ③ infer_history.json 에서 직전 실행을 찾아
                                 #    비교 뷰어 자동 생성 (개선 초록/악화 빨강, 변화 큰 샘플 우선)
results/qwen3vl/
├── infer_latest.html            # 최신 추론 뷰어 (고정 이름, 매회 덮어씀)
├── compare_latest.html          # 최신 vs 직전 비교 뷰어 (고정 이름)
├── infer_step{N}_n100.json      # step 별 데이터 (이력)
└── infer_history.json           # 실행 이력 manifest
```

구현은 [tools/infer_fashion_batch.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/tools/infer_fashion_batch.py)(추론+이력 관리)와
[tools/compare_epochs_html.py 의 build_compare()](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/tools/compare_epochs_html.py#L47)(비교 뷰어 생성)에 있다.
9.5 의 에폭별 표는 전부 이 도구의 산출물이다. HTML 은 이미지 base64 내장 단일 파일이라
서버에서 내려받기만 하면 열린다(폐쇄망 HTTP 차단 환경 대응).

---

## 9.8 정리

| 선택 | 근거 | 상태 |
|---|---|---|
| roi 제거 출력 | 아이템 수=그룹 수 전수 일치 → 모호성 없음 | 적용 (roi 판은 예비 보관) |
| prefix 단일 소스 | 학습·추론 프롬프트 불일치 사고 차단 | 적용 |
| max_pixels 320토큰 | 속성 인식은 고해상도 불필요 | 적용 (검출 확장 시 재검토) |
| 유효배치 256 · lr 1e-5 | 오피셜 레시피 정렬 | 적용 |
| SDPA 마스크 `.to(q.dtype)` | bf16 + 패딩 배치에서만 드러나는 크래시 | 적용 |
| generate() 배치 패치 | 프리필 pad-로짓 · 동시 EOS 요구 | 적용 (단건 대조 검증) |
| 3에폭 중단 | loss 미세 하강 ≠ 정확도 개선 (9.5) | 적용 |

### 남은 숙제

| # | 항목 | 비고 |
|---|---|---|
| 1 | **eval 10,000장 전체 평가** | 100장의 ±3%p 는 노이즈 — 공식 수치는 1만 장으로 |
| 2 | vision lr 차등(2e-6) | [optimizer_param_groups 훅](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/pierrot/core/engine.py#L83)을 qwen3vl 에 구현 |
| 3 | 대표 카테고리 엄격 판정 | 동의어 교집합은 "dress 겹침"만으로 ✓ — 첫 값 일치 기준 병행 |
| 4 | roi 판(`data/fashion`) 학습 | 속성+박스 동시 예측([include_roi](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/tools/build_fashion_jsonl.py#L31)), max_pixels 상향 검토 |
| 5 | vLLM 서빙 | 대량/서비스 추론은 continuous batching 으로 |
