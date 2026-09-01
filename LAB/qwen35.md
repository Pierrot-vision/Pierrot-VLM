# Qwen3.5-4B — 코드 기반 전체 플로우

Pierrot-VLM-Lab 의 Qwen3.5 구현을 **실제 코드 흐름** 그대로 따라가는 문서다.
학습 1스텝과 추론 1회가 코드에서 어떤 함수를 어떤 순서로 통과하는지 추적한다.

> 이 저장소는 **추론 배포본**이다. 학습 스크립트(`training/`)·하이퍼파라미터(`args/`)·
> 데이터 빌더는 들어 있지 않으므로, 본문에서 그런 파일을 가리키는 링크는 학습 저장소
> [Pierrot-VLM-Lab](https://github.com/Pierrot-vision/Pierrot-VLM-Lab) 로 연결된다.

> 백본 요약: **동적 해상도 ViT(Qwen3-VL 계약과 동일, DeepStack 없음) → 패치 머저 →
> 하이브리드 언어 디코더(Gated DeltaNet 3 : Gated Attention 1)**.
> 다른 네 모델과 마찬가지로 **모델 본체가 순수 PyTorch 스크래치 구현**이다 —
> Gated DeltaNet(청크 병렬식/순차 점화식), 출력 게이트 어텐션, 부분 회전 M-RoPE,
> zero-centered RMSNorm 을 [modeling/text.py](../pierrot/models/qwen35/modeling/text.py) 에
> 전부 다시 썼고, 공식 체크포인트와 텐서 키·shape 가 1:1 로 일치한다(mtp 제외 723개 검증).
> 전처리·데이터·학습 인프라는 Qwen3-VL 과 같은 입력 계약이라 그대로 재사용한다.
> Qwen3.5 는 텍스트/비전 라인이 분리돼 있던 Qwen3 와 달리 **처음부터 멀티모달 단일 라인**이다.

---

## 0. 파일 지도

```
Pierrot_VLM/
├── args/qwen35.py                  # 하이퍼파라미터 단일 소스 (PARAMS dict → 평탄한 args)
├── training/train_qwen35.py        # 학습 진입점
├── infer/infer_qwen35.py           # 추론 진입점
└── pierrot/
    ├── core/                          # 모델-비의존 학습 인프라 (다른 모델과 공유)
    │   ├── registry.py                #   ModelSpec 인터페이스 + 레지스트리
    │   ├── engine.py                  #   Trainer (Accelerate 학습 루프)
    │   └── scheduler.py               #   warmup + cosine LR
    ├── data/                          # 공용 데이터 소스 (모델 간 공유)
    │   ├── jsonl.py                   #   JsonlDataset ({image,prefix,suffix})
    │   └── coco.py                    #   CocoDetectionSource
    └── models/qwen35/
        ├── config.py                  # Qwen35VisionConfig / Qwen35TextConfig / Qwen35Config
        ├── modeling/
        │   ├── vision.py              #   동적 해상도 ViT — qwen3vl 블록 재사용, DeepStack 없음
        │   ├── text.py                #   ★ 하이브리드 디코더 스크래치 구현 (이 문서의 주인공)
        │   └── qwen35.py              #   최상위: 병합(masked_scatter) + M-RoPE 위치 + loss + generate(상태 캐시)
        ├── processor.py               # Qwen3-VL 프로세서 상속 (동적 해상도 패치화 + ChatML + 라벨)
        ├── dataset.py                 # qwen3vl 데이터셋/collate 재수출 (동일 입력 계약)
        ├── weights.py                 # HF Hub 다운로드 + safetensors 로드 (raw config · mtp.* 무시)
        └── spec.py                    # 레지스트리 어댑터 (@register_model "qwen35")
```

두 종류의 "config" 를 구분하는 것이 이해의 핵심이다:

| 종류 | 클래스 | 어디서 오나 | 무엇 |
|---|---|---|---|
| **실험 설정** | `args` (평탄 네임스페이스) | [args/qwen35.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/args/qwen35.py) | lr, batch, epochs, 어느 체크포인트, 동결, **max_pixels** |
| **모델 구조** | `Qwen35Config` 등 | 체크포인트 `config.json` (raw) | 하이브리드 레이어 배치, GDN 헤드 수, partial RoPE, 비전 격자... (자동) |

> **Qwen3-VL 과의 재사용 관계**: **입력 계약이 같다** — `input_ids / attention_mask /
> labels / pixel_values(패치 패킹) / image_grid_thw`. 그래서 processor·dataset·collate 와
> 비전 빌딩 블록(패치 임베딩·bilinear 위치보간·비전 블록·머저)은 qwen3vl 것을
> 상속/재사용하고, **새로 구현한 것은 디코더 전체와 config/weights/최상위 결합**이다.
> config 복원은 raw JSON 파싱만 쓰므로(AutoConfig 불필요) Qwen3.5 를 아직 모르는
> transformers(<5.1) 환경에서도 이 구현이 그대로 동작한다.

---

## 1. 학습 전체 플로우 (training/train_qwen35.py 진입 → 저장)

```
python training/train_qwen35.py
   │
   ├─(a) from args.qwen35 import args      # PARAMS → 평탄한 args (단일 소스)
   ├─(b) _preflight(args)                  # 다운로드 前 데이터/이미지 경로 검사 + 픽셀 예산 출력
   ├─(c) spec = get_model_spec('qwen35')   # 레지스트리에서 Qwen35Spec 선택
   ├─(d) model, processor = spec.build(args)     # 아래 2절
   ├─(e) train_ds = spec.build_dataset(args,'train',processor)
   │     collate  = spec.collate_fn(processor,args)
   ├─(f) DataLoader(train_ds, collate_fn=collate, ...)
   └─(g) Trainer(model, loader, args, processor=...).fit()   # Accelerate 루프(다른 모델과 공유)
             └ 저장: output_dir/final/{model.pt, config.json(Qwen35Config asdict), tokenizer, qwen3vl_preprocessor.json}
```

---

## 2. 모델 생성 — `spec.build(args)`

[spec.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/pierrot/models/qwen35/spec.py) 의 흐름 (qwen3vl 과 동일 패턴):

```
build(args)
   ├─ pretrained 있음 → weights.load_pretrained(...)        # 아래
   │     ├ resolve_model_dir: 로컬 경로 or HF Hub snapshot_download
   │     ├ config_from_json: raw config.json → Qwen35Config
   │     │    ├ text_config.rope_parameters{rope_theta, partial_rotary_factor,
   │     │    │                             mrope_section, mrope_interleaved} → 평탄 필드 승격
   │     │    └ layer_types 그대로 (없으면 full_attention_interval 로 유도)
   │     ├ build_processor: sidecar/공식 preprocessor 픽셀 예산 상속 (qwen3vl 규약)
   │     ├ Qwen35ForConditionalGeneration(config)            # 스크래치 모델
   │     ├ safetensors 적재 → mtp.* 제거 → load_state_dict(strict=False)
   │     │    └ mtp.* = multi-token prediction 보조 헤드(투기적 디코딩용) — 본 모델과 무관
   │     ├ lm_head 누락 시 tie(임베딩 공유) → _verify_load(엄격: 그 외 missing/unexpected 예외)
   │     └ dtype/device 이동
   ├─ pretrained 없음 → model_extra 기반 config 로 랜덤 초기화(스크래치 학습)
   ├─ _apply_freezing: freeze_vision → visual.{patch_embed,pos_embed,blocks}
   │                   freeze_projector → visual.merger (DeepStack 머저 없음)
   └─ gradient_checkpointing_enable()     # 비전 블록 + 디코더 레이어 체크포인팅
```

---

## 3. 데이터 → 배치 플로우

**이미지 전처리·collate 는 qwen3vl 재사용**([dataset.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/pierrot/models/qwen35/dataset.py) 는
재수출 9줄)이고, [processor.py](../pierrot/models/qwen35/processor.py) 는 상속하되
**encode_one 을 오버라이드**한다 — Qwen3.5 공식 템플릿의 **thinking 접두** 때문이다.
자세한 공통 동작은 [qwen3vl.md 3절](qwen3vl.md) 참조. 요점만:

```json
{"image": "images/cat.jpg", "prefix": "질문/지시문", "suffix": "정답"}
```

```
① smart_resize(h,w, factor=16×2=32, min_pixels, max_pixels)   # 종횡비 유지, 32 배수
   → 패치 (n_patches, patch_dim=1536) + 격자 grid=(1, h/16, w/16)
② <|vision_start|> + <|image_pad|>×(t·h·w/4) + <|vision_end|>
③ ChatML: <|im_start|>user\n{이미지}{prefix}<|im_end|>\n<|im_start|>assistant\n{think접두}{suffix}<|im_end|>
   think접두 = enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n"   # ★ Qwen3.5 공식 템플릿
④ 라벨: user/이미지/think접두 = -100, {suffix}+끝 <|im_end|> 만 정답 id (suffix-only 손실)
⑤ collate: input_ids/attention_mask/labels (B,Lmax) 우측 패딩,
           pixel_values (총패치수, 1536) 패딩 없이 패킹, image_grid_thw (이미지수, 3)
```

> **★ thinking 접두는 프롬프트 쪽(-100)이다.** 추론 때도 같은 접두를 템플릿이 제공하므로
> (공식 add_generation_prompt 와 동일) 모델은 답변 토큰만 배우고, 학습/추론 프롬프트 분포가
> 정확히 일치한다. 기본 `enable_thinking=False`(빈 사고 블록) — suffix 가 사고 과정 없는
> 정답(검출 좌표 등)인 Pierrot 파인튜닝에 맞는 모드다. 이 값은 args 단일 소스에서 오고
> sidecar 에 동봉돼 추론이 자동 상속한다(픽셀 예산과 같은 원리).

특수 토큰 id 만 다르다 — Qwen3.5 는 어휘가 커져서(248,320) id 대역이 바뀌었고,
프로세서가 토크나이저에서 실제 id 를 읽어 config 에 반영한다:

| 토큰 | Qwen3-VL (vocab 151,936) | Qwen3.5 (vocab 248,320) |
|---|---|---|
| `<\|vision_start\|>` / `<\|vision_end\|>` | 151652 / 151653 | 248053 / 248054 |
| `<\|image_pad\|>` / `<\|video_pad\|>` | 151655 / 151656 | 248056 / 248057 |

> **★ max_pixels 는 여전히 가장 강력한 손잡이다.** 이미지 토큰 수 ≈ max_pixels/1024.
> args 기본은 Qwen3-VL 과 같은 327,680(이미지당 최대 320 토큰). 학습·추론 기본값이 모두
> [args/qwen35.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/args/qwen35.py) 의 PARAMS 에서 오므로 단일 소스로 일치한다
> (추론 CLI 의 `--max-pixels` 기본값도 PARAMS). 학습 산출물에는 sidecar
> `qwen3vl_preprocessor.json` 이 동봉돼 재로드 시 자동 상속된다.

---

## 4. 모델 forward — `Qwen35ForConditionalGeneration.forward`

### 4.1 비전 타워 ([modeling/vision.py](../pierrot/models/qwen35/modeling/vision.py)) — Qwen3-VL ViT 계약, DeepStack 제거

```
patch_embed: (S, 1536) → (S, 1024)          # patch16 · temporal2 · 24블록 · 16헤드
위치 임베딩: 48×48(2304) 테이블 → 격자 (h,w) bilinear 보간
패치 머저: 2×2 이웃을 접어 (S/4, 4096) → 선형 → (S/4, 2560=언어 hidden)
```

블록 단위까지 Qwen3-VL 과 동일해 qwen3vl 의 스크래치 빌딩 블록을 import 로 재사용하고,
이 파일은 **DeepStack 을 뺀 타워 조립**만 담당한다 — 중간층 특징을 뽑는 별도 머저가
없고, 비전 타워는 최종 머저 출력 하나만 낸다. 병합은 `<|image_pad|>` 자리 교체
(masked_scatter) 한 번으로 끝난다.

### 4.2 하이브리드 언어 디코더 ([modeling/text.py](../pierrot/models/qwen35/modeling/text.py)) — 8 × (3 GDN → 1 Gated Attention)

#### 하이브리드 배치 — 4층마다 하나만 full attention

**"Gated DeltaNet 3:1 하이브리드"란 디코더 층 4개 중 3개는 Gated DeltaNet, 1개만 보통
어텐션이라는 뜻이다.**

왜 섞는가. 보통 트랜스포머 디코더는 **모든 층이 self-attention** 인데 대가가 둘이다.

- 연산이 시퀀스 길이의 **제곱**(O(T²))
- 생성 중 **KV 캐시가 길이에 비례해 계속 불어난다** — 문서 한 페이지처럼 긴 입력에서 부담이 크다

**Gated DeltaNet**(GDN)은 linear attention 계열이라 과거를 전부 들고 있지 않고 **고정 크기
상태 하나**(헤드별 128×128 행렬)로 요약해 굴린다. 길이와 무관하게 메모리가 일정하고 연산도
O(T) 다. 대신 "앞쪽 특정 토큰을 정확히 집어오는" 일은 softmax attention 보다 약하다.

그래서 **대부분을 싼 층으로 채우고, 정밀한 참조가 필요한 자리만 남긴다.** 그 비율이 3:1 이다.

```
층  0~7   · · · F · · · F
층  8~15  · · · F · · · F        ·  Gated DeltaNet (linear)  24개
층 16~23  · · · F · · · F        F  full attention            8개
층 24~31  · · · F · · · F        24 : 8 = 3 : 1
                                 F 위치 = 층 3, 7, 11, 15, 19, 23, 27, 31
```

코드에는 비율이 아니라 **간격**으로 적혀 있고, `layer_types` 를 명시하지 않으면 거기서
유도된다 ([config.py:95-117](../pierrot/models/qwen35/config.py#L95-L117)):

```python
full_attention_interval: int = 4          # 4번째마다 full → 자동으로 3:1
layer_types: Optional[List[str]] = None   # 명시하면 그대로, 없으면 아래로 유도

self.layer_types = [
    "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
    for i in range(self.num_hidden_layers)
]
```

**실질적인 결과는 생성 캐시가 두 종류가 된다는 것이다**
([text.py:556-560](../pierrot/models/qwen35/modeling/text.py#L556-L560)):

```python
def new_cache(self) -> List[Dict]:
    return [
        {"key": None, "value": None}          # full attention 층 → KV 캐시 (길이만큼 쌓인다)
        if t == "full_attention" else
        {"conv": None, "recurrent": None}     # GDN 층 → 고정 크기 상태 (안 늘어난다)
        for t in self.cfg.layer_types
    ]
```

32층 중 **24층이 아래쪽**이라, 긴 문서를 생성해도 캐시 메모리가 일반 트랜스포머보다 훨씬
천천히 는다. 이 배치는 Qwen3.5 공식 설정을 그대로 따랐다.

#### 두 종류 층의 내부

32개 레이어의 `layer_types` 가 `[linear, linear, linear, full] × 8`
(`full_attention_interval=4`) — **24층은 Gated DeltaNet, 8층만 softmax attention** 이다.

**Gated DeltaNet 층** (`Qwen35GatedDeltaNet`, 시퀀스 길이에 O(T)):

```
입력 → in_proj_qkv (2560→8192) → depthwise causal Conv1d(kernel=4) + silu → q,k,v
       in_proj_z → z(출력 게이트) · in_proj_b → β · in_proj_a → a
상태 S (헤드별 128×128 행렬) 를 토큰마다 갱신:
   S_t = g_t · S_{t-1} + β_t · k_t ⊗ (v_t − S_{t-1}ᵀ k_t)
         └ gated decay: g=exp(−exp(A_log)·softplus(a+dt_bias))   └ delta rule(차이만 기록)
출력 o_t = S_tᵀ q_t → RMSNormGated(o, z)(정규화 후 silu(z) 게이트) → out_proj
헤드: QK 16개×128(V 수에 맞춰 2배 복제) · V 32개×128 — q/k 는 L2 정규화(상태 폭주 방지)
```

같은 수식을 **두 방식**으로 구현했고 등가성을 테스트로 고정했다:

| 함수 | 언제 | 방식 |
|---|---|---|
| `chunk_gated_delta_rule` | 학습·프리필 | 64토큰 청크 병렬식 — 청크 안은 행렬식, 청크 사이는 상태 전달. 청크 내 순차 의존은 하삼각 역행렬(전진대입)로 한 번에 해소 |
| `recurrent_gated_delta_rule` | 디코드(T=1) | 점화식 그대로 |

**Gated Attention 층** (`Qwen35TextAttention`, softmax):

- 표준 GQA — Q 16헤드 / KV 4헤드, head_dim **256**, QK-Norm(zero-centered) 포함.
- **출력 게이트**(`attn_output_gate`) — `q_proj` 출력이 2배(헤드마다 쿼리‖게이트).
  어텐션 결과에 `sigmoid(gate)` 를 곱한 뒤 `o_proj`. attention sink·거대 활성값 완화 장치.
- **partial M-RoPE** — `partial_rotary_factor=0.25`: head_dim 256 중 **64차원만 회전**,
  나머지는 통과. `rope_theta=1e7`, 3축 배분 `mrope_section=[11,11,10]` 은 회전 주파수
  32개 안에서 interleaved. 이미지 격자 좌표 → 3축 위치는 `get_rope_index` 가 만든다
  (qwen3vl 과 동일 규칙: 텍스트는 3축 동시 증가, 이미지는 (t,h,w) 격자, 폭 max(h,w)/m).

**zero-centered RMSNorm** (`Qwen35RMSNorm`) — 블록/최종/QK norm 전부:

```python
out = rms_normalize(x) * (1.0 + weight)      # weight 초기값 0, 체크포인트도 0 근처
```

가중치를 0 중심으로 저장하는 규약(weight decay 와 궁합). **표준 RMSNorm 으로 로드하면
오류 없이 로드되지만 출력이 전부 틀리는** 가장 조용한 함정이라 별도 테스트로 고정했다.
(DeltaNet 내부 `RMSNormGated` 는 표준 1 중심.)

### 4.3 손실 (suffix-only, 메모리 절약형)

qwen3vl 과 동일 — labels 를 1칸 시프트한 뒤 **-100 이 아닌 위치의 hidden 만** lm_head 에
통과시켜 CE 를 계산한다. vocab 이 248,320 이라 전체 (B,T,V) 로짓을 만들면 메모리가
크게 뛰므로 이 트릭의 이득이 특히 크다(그래도 `batch_size=1 + grad_accum=16` 이 args 기본).

---

## 5. 학습 루프 — `Trainer.fit` (Accelerate)

네 선배 모델과 **완전히 동일한 엔진**([engine.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/pierrot/core/engine.py))이다.
`forward(**batch) → {"loss"}` 규약을 지키므로 grad accum / bf16 / DDP·FSDP /
체크포인트 재개 / cosine LR / NaN 가드가 전부 그대로 동작한다. 저장 시:

```
output_dir/final/
├── model.pt                     # 스크래치 모델 state_dict (model.visual.*, model.language_model.*)
├── config.json                  # Qwen35Config asdict (text/vision 하위 dict 포함)
├── tokenizer.json ...           # 토크나이저
└── qwen3vl_preprocessor.json    # 전처리 sidecar (min/max_pixels, patch/merge, system_prompt)
```

재로드는 [weights.py](../pierrot/models/qwen35/weights.py) `config_from_json` 이 공식/산출물
양쪽 형태를 모두 읽고, `_load_state_dict` 가 safetensors(공식) 또는 model.pt(산출물)를
적재한다 — 로드 검증이 엄격해(missing/unexpected 예외) 조용한 랜덤 초기화가 없다.

---

## 6. 추론 전체 플로우 (infer/infer_qwen35.py)

```
python infer/infer_qwen35.py --model Qwen/Qwen3.5-4B --image cat.jpg --prompt "Describe this image."
   │
   ├─ load_pretrained(model, min/max_pixels=PARAMS 기본)   # ★ 학습과 같은 단일 소스
   ├─ inputs = processor([image], [prompt])    # suffix 없음 → labels 없음
   ├─ model.generate(..., eos_token_id=processor.eos_token_id)   # 스크래치 generate (아래)
   └─ 프롬프트 뒤 새 토큰만 tokenizer.decode → 출력
```

`generate` 는 **하이브리드 상태 캐시**를 쓴다 (`Qwen35TextModel.new_cache()`):

```
프리필: 캐시를 미리 만들어 넘기고(★ qwen3vl 과 다른 점 — linear 층이 프리필 중에
        상태를 저장할 자리가 필요) 이미지 병합된 프롬프트를 한 번에 처리
   full attention 8층  → KV 캐시 (B, 4, T, 256)×2         # 시퀀스 길이에 비례
   GDN 24층            → conv (B, 8192, 3) + recurrent (B, 32, 128, 128)
                          # ★ 고정 크기 — 긴 문맥에서 캐시 메모리 우위
디코드: 새 토큰 1개씩 — full 층은 KV concat, GDN 층은 conv 한 칸 + 점화식 한 스텝.
        위치는 next_pos 부터 3축 동시 증가. greedy(do_sample=False), eos 에서 종료.
배치 안전: 첫 로짓은 샘플별 '실제 마지막 토큰' 위치에서 뽑고(우측 패딩 대응),
        EOS 는 샘플별 finished 로 추적 — 먼저 끝난 샘플은 EOS 로 채운다(qwen3vl 동일).
DeltaNet pad 면역(★ Qwen3.5 고유): 우측 pad 가 프리필 상태를 오염시키는 세 경로를
        전부 차단한다 — ① 입력 0 마스킹 ② pad 스텝 감쇠 차단(g=0; 입력을 지워도
        g 는 dt_bias 때문에 0 이 아니다) ③ conv 후 재마스킹(인과 conv 가 pad 위치에서
        직전 실토큰 window 를 보는 누수) + conv 캐시는 샘플별 유효 길이 기준 gather.
        → 패딩 배치의 conv/recurrent 상태가 단독(B=1) 실행과 정확히 일치한다.
```

> **★ 전처리 일치**: `--max-pixels` 를 학습과 똑같이 맞춰야 이미지 토큰 수가 같아진다.
> CLI 기본값이 PARAMS 이므로 args/qwen35.py 를 바꿨다면 추론도 자동으로 따라온다.
> 파인튜닝 산출물 디렉토리를 `--model` 로 주면 config.json + model.pt 로 그대로 복원된다.

---

## 7. 스펙 요약 (Qwen3.5-4B 기준, config.json)

```
이미지  smart_resize → (S, 1536)                # S=총 패치 수(이미지마다 다름)
  └ 비전 24블록 (DeepStack 없음)   → (S, 1024)
  └ 패치 머저(m=2)                 → (S/4, 2560)  ─┐ 병합: <|image_pad|> 자리 교체
텍스트  input_ids (B, T) → embed → (B, T, 2560) ─┘
                   → 하이브리드 32층 [GDN×3 → Attn×1]×8 → (B, T, 2560)
                   → lm_head(tie) → logits (B, T, 248320)
                   → CE(ignore=-100, suffix 위치만) → loss (스칼라)
```

| 이름 | 값 (Qwen3.5-4B) |
|---|---|
| 비전 hidden / depth / heads | 1024 / 24 / 16 · patch16 · merge2 · **DeepStack 없음** |
| 위치 임베딩 격자(보간 소스) | 48×48 (2304) → bilinear |
| 언어 hidden / 레이어 | 2560 / 32 = **[GDN, GDN, GDN, Attn] × 8** |
| Gated DeltaNet 헤드 | QK 16×128 / V 32×128 · conv kernel 4 |
| Gated Attention 헤드 (Q/KV) | 16 / 4 (GQA) · head_dim **256** · **출력 게이트** |
| RoPE | partial 0.25 (64/256차원만 회전) · rope_theta 1e7 · mrope [11,11,10] |
| RMSNorm | **zero-centered** (1 + weight) |
| vocab / 네이티브 문맥 | **248,320** / 262,144 토큰 |
| 이미지 토큰 수 | **가변** ≈ max_pixels/1024 (args 기본 → 최대 320) |

> 크기 변형은 config.json 이 처리한다 — args 의 `pretrained` 만 바꾸면 된다
> (layer_types·헤드 수·hidden 전부 raw 파싱으로 복원).

---

## 8. 한눈에 보는 전체 그림

```
                 args/qwen35.py (PARAMS 단일 소스)
                        │  args.lr, args.pretrained, max_pixels, ...
        ┌───────────────┴──────────────────┐
        ▼                                   ▼
   spec.build(args)                    Trainer(args)  ── Accelerate 루프(공유 엔진)
   ├ weights.load_pretrained            └ for batch: model(**batch)["loss"]
   │   ├ raw config.json → Qwen35Config    ├ backward / clip / step
   │   ├ safetensors 적재 (mtp.* 무시)      └ save_pretrained(final)
   │   └ 엄격 로드 검증 + tie                     │
   ├ Qwen35ForConditionalGeneration (스크래치)     ▼
   └ Qwen35Processor (= Qwen3VLProcessor)   output_dir/final/{model.pt,
        │                                    config.json(Qwen35Config),
        │                                    tokenizer, qwen3vl_preprocessor.json}
        └── DataLoader(collate=패치 패킹, qwen3vl 재사용) ──► infer/infer_qwen35.py:
                                                              load_pretrained → 스크래치 generate
```

**핵심 5가지** (선배 네 모델 대비 Qwen3.5 구현의 특징):
1. **하이브리드 디코더 스크래치** — Gated DeltaNet(청크/순차 두 구현, 등가성 테스트로 고정)
   24층 + 출력 게이트 attention 8층을 순수 PyTorch 로 재현 (4.2)
2. **고정 크기 상태 캐시** — GDN 층은 시퀀스가 길어져도 conv/recurrent 상태 크기가 일정,
   KV 캐시는 full 8층에만 있다 (6)
3. **DeepStack 제거·partial RoPE** — Qwen3-VL 대비 단순해진 비전-언어 결합, RoPE 는
   head_dim 의 25%만 회전 (4.1·4.2)
4. **zero-centered RMSNorm** — (1 + weight) 규약. 표준 RMSNorm 으로 로드하면 조용히
   틀리는 함정이라 테스트로 고정 (4.2)
5. **입력 계약 재사용** — qwen3vl 의 동적 해상도 전처리/패치 패킹/라벨 마스킹/비전 블록을
   그대로 상속, 특수 토큰 id 는 토크나이저에서 자동 확정 (3·4.1)

> **검증**: [tests/test_qwen35.py](https://github.com/Pierrot-vision/Pierrot-VLM-Lab/blob/main/tests/test_qwen35.py) 30개(다운로드 불필요) —
> 청크↔순차 delta rule 등가성, conv 캐시 = 프리필 일치, 캐시 디코드 = full forward 일치,
> 게이트 어텐션/conv shape(공식 체크포인트 로드 호환), zero-centered norm, 부분 RoPE 통과,
> thinking 접두 위치/마스킹, 우측 패딩 배치의 첫 토큰 위치, **pad 면역(상태 수준 +
> 패딩 배치 다중 토큰 생성 = 단독 실행)**, 샘플별 EOS 채움,
> loss/generate 계약, 공식 config.json(rope_parameters) 파싱, 저장/재로드.
> 추가로 `Qwen/Qwen3.5-4B` 공식 체크포인트의 safetensors 메타데이터(키+shape, 가중치
> 다운로드 없음)와 대조해 **mtp 제외 723개 텐서 전부 1:1 일치**를 확인했다(2026-07-28).
> 공식 구현 대비 수치 parity 는 [tools/parity_qwen35.py](../tools/parity_qwen35.py) (opt-in,
> transformers>=5.1 필요 — 미달 환경에선 skip)로 전처리/비전/로짓/greedy 4단계를 검증한다.
> 남은 것: transformers>=5.1 환경에서 parity 실행, 실데이터 파인튜닝.
