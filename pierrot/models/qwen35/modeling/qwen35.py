"""Qwen3.5 최상위 모델 (스크래치, 추론 전용).

구성: 동적 해상도 비전 타워(visual) + 하이브리드 디코더(language_model) + lm_head.
모듈 이름은 HF Qwen3.5 체크포인트 키와 일치시켜 공개 가중치가
load_state_dict(strict=False) 로 바로 들어온다:
    model.visual.* / model.language_model.* / lm_head.weight(=임베딩 tie)
(공식 체크포인트의 mtp.* — multi-token prediction 보조 헤드 — 는 본 모델과 무관해
weights.load_pretrained 가 로드 전에 걸러낸다.)

이미지 병합·M-RoPE 위치 계산은 qwen3vl 과 동일한 규칙이다(같은 전처리 계약).
DeepStack 이 없으므로 비전 타워는 머저 출력 하나만 내고 재주입 경로도 없다.

표준 causal LM 생성이다 — forward 는 로짓만 내고, generate 가 KV 캐시로 autoregressive
하게 이어 붙인다. 손실·활성화 체크포인팅 등 학습 경로는 학습 저장소 Pierrot-VLM-Lab 쪽에 있다.

생성 캐시: full attention 레이어는 KV 캐시, linear attention(Gated DeltaNet)
레이어는 conv/recurrent **고정 크기 상태** 를 갱신한다 — 프리필에서
language_model.new_cache() 로 만든 리스트를 디코드 내내 재사용한다.

텐서 차원 표기:
    B = 배치, T = 시퀀스 길이, D = text hidden, V = vocab
    S = 총 패치 수, m = spatial_merge_size, S/m² = 총 이미지 토큰 수
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Qwen35Config
from .text import Qwen35TextModel
from .vision import Qwen35VisionModel


class Qwen35Model(nn.Module):
    """비전 타워 + 하이브리드 디코더 묶음 (HF 키 model.* 에 대응)."""

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config         = config
        self.visual         = Qwen35VisionModel(config.vision_config)
        self.language_model = Qwen35TextModel(config.text_config)


class Qwen35ForConditionalGeneration(nn.Module):
    """Qwen3.5: 동적 해상도 ViT + Gated DeltaNet 하이브리드 디코더 결합 VLM."""

    # ------------------------------------------------------------------ #
    # model(비전/언어)과 lm_head 를 조립한다. 이름은 HF 키와 일치.
    # ------------------------------------------------------------------ #
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config         = config
        self.model          = Qwen35Model(config)
        self.lm_head        = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.image_token_id = config.image_token_id
        self.merge_size     = config.vision_config.spatial_merge_size

    # ------------------------------------------------------------------ #
    # lm_head 가중치를 텍스트 임베딩과 공유(weight tying).
    # Qwen3.5-4B 는 tie 가 기본이라 체크포인트에 lm_head.weight 가 없다.
    # ------------------------------------------------------------------ #
    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.language_model.embed_tokens

    # ------------------------------------------------------------------ #
    # 패킹된 패치 시퀀스를 인코딩한다: (S, patch_dim) → (S/m², D).
    # ------------------------------------------------------------------ #
    def _encode_images(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor,
                       dtype: torch.dtype) -> torch.Tensor:
        return self.model.visual(pixel_values.to(dtype), image_grid_thw)

    # ------------------------------------------------------------------ #
    # 텍스트 임베딩의 <|image_pad|> 자리를 이미지 임베딩으로 교체한다(qwen3vl 동일).
    # ------------------------------------------------------------------ #
    def _merge_image_features(self, input_ids, inputs_embeds, image_embeds):
        mask     = input_ids == self.image_token_id
        n_tokens = int(mask.sum())
        if n_tokens != image_embeds.shape[0]:
            raise ValueError(
                f"<|image_pad|> 토큰 수({n_tokens}) != 이미지 임베딩 수({image_embeds.shape[0]}). "
                f"프로세서의 grid_thw 와 모델의 spatial_merge_size 가 일치하는지 확인하세요."
            )
        merged = inputs_embeds.masked_scatter(
            mask.unsqueeze(-1), image_embeds.to(inputs_embeds.dtype)
        )
        return merged, mask

    # ------------------------------------------------------------------ #
    # M-RoPE 용 3축 position_ids (3, B, T) 와 각 샘플의 "다음 위치"(B,)를 만든다.
    # 텍스트는 세 축 동시 증가, 이미지는 격자 좌표, 이미지 폭은 max(h,w)/m —
    # qwen3vl 과 동일한 규칙이다(공식 Qwen3.5 도 같은 compute_3d_position_ids).
    # ------------------------------------------------------------------ #
    def get_rope_index(self, input_ids: torch.Tensor, image_grid_thw: Optional[torch.Tensor],
                       attention_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        device   = input_ids.device
        B, T     = input_ids.shape
        m        = self.merge_size
        position_ids = torch.zeros(3, B, T, dtype=torch.long, device=device)
        next_pos     = torch.zeros(B, dtype=torch.long, device=device)

        grids    = image_grid_thw.tolist() if image_grid_thw is not None else []
        grid_idx = 0

        for b in range(B):
            valid = attention_mask[b].bool() if attention_mask is not None else torch.ones(T, dtype=torch.bool, device=device)
            ids   = input_ids[b][valid]
            n     = ids.numel()
            is_img = (ids == self.image_token_id)

            segments: List[torch.Tensor] = []
            pos, i = 0, 0
            while i < n:
                if is_img[i]:
                    t, h, w = grids[grid_idx]
                    grid_idx += 1
                    lh, lw   = h // m, w // m
                    length   = t * lh * lw
                    tt, hh, ww = torch.meshgrid(
                        torch.arange(t, device=device),
                        torch.arange(lh, device=device),
                        torch.arange(lw, device=device),
                        indexing="ij",
                    )
                    segments.append(torch.stack([tt, hh, ww], dim=0).reshape(3, -1) + pos)
                    pos += max(h, w) // m
                    i   += length
                else:
                    j = i
                    while j < n and not is_img[j]:
                        j += 1
                    length = j - i
                    segments.append(
                        torch.arange(length, device=device).view(1, -1).expand(3, -1) + pos
                    )
                    pos += length
                    i    = j

            llm_positions = torch.cat(segments, dim=1)
            if llm_positions.shape[1] != n:
                raise ValueError(
                    f"M-RoPE 위치 길이({llm_positions.shape[1]}) != 유효 토큰 수({n}). "
                    f"image_grid_thw 와 <|image_pad|> 배치가 어긋났습니다."
                )
            position_ids[:, b, valid] = llm_positions
            next_pos[b] = int(llm_positions.max()) + 1

        return position_ids, next_pos

    # ------------------------------------------------------------------ #
    # 이미지 인코딩 → 병합 → M-RoPE 위치까지, forward/generate 공통 준비 단계.
    # ------------------------------------------------------------------ #
    def _prepare_inputs(self, input_ids, pixel_values, image_grid_thw, attention_mask):
        inputs_embeds = self.get_input_embeddings()(input_ids)                 # (B, T, D)

        if pixel_values is not None:
            image_embeds = self._encode_images(pixel_values, image_grid_thw, inputs_embeds.dtype)   # (S/m², D)
            inputs_embeds, _ = self._merge_image_features(input_ids, inputs_embeds, image_embeds)   # (B, T, D) 자리 교체

        position_ids, next_pos = self.get_rope_index(input_ids, image_grid_thw, attention_mask)     # (3, B, T), (B,)
        return inputs_embeds, position_ids, next_pos

    # ------------------------------------------------------------------ #
    # 순전파. 이미지+텍스트를 받아 로짓을 낸다.
    #   ① 텍스트 임베딩 → ② 비전 인코딩 → ③ <|image_pad|> 자리에 병합
    #   → ④ M-RoPE position_ids → ⑤ 하이브리드 디코더 → hidden.
    # logits_to_keep>0 이면 마지막 N 위치의 로짓만 만든다(생성 프리필에서 (B,T,V)
    # 전체를 만들지 않아 메모리를 아낀다).
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        inputs_embeds, position_ids, _ = self._prepare_inputs(
            input_ids, pixel_values, image_grid_thw, attention_mask
        )
        hidden, _ = self.model.language_model(
            inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
        )                                                                      # (B, T, D)

        if logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:, :]
        return {"logits": self.lm_head(hidden)}

    # ------------------------------------------------------------------ #
    # 상태 캐시 기반 autoregressive 생성 (qwen3vl 과 같은 계약).
    #   - 프리필: new_cache() 로 하이브리드 캐시를 만들어 프롬프트를 한 번에 처리.
    #     full attention 레이어는 KV 를 쌓고, linear attention 레이어는 conv/
    #     recurrent 상태를 저장한다(청크식이 마지막 상태를 돌려준다).
    #   - 디코드: 새 토큰 1개만 임베딩. full attention 은 캐시 전체 참조, linear
    #     attention 은 점화식 한 스텝(상태 갱신)이다. 위치는 next_pos 부터 세 축
    #     동시 증가.
    # do_sample 이면 top-k→top-p 샘플링, 아니면 greedy. <eos> 만나면 종료.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,                                    # 0/None = 비활성(공식 기본은 20)
        eos_token_id=None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens 는 0 이상이어야 합니다: {max_new_tokens}")
        if do_sample:
            if temperature <= 0:
                raise ValueError(f"do_sample 에서 temperature 는 양수여야 합니다: {temperature}")
            if not (0.0 < top_p <= 1.0):
                raise ValueError(f"top_p 는 (0, 1] 범위여야 합니다: {top_p}")
            if top_k is not None and top_k < 0:
                raise ValueError(f"top_k 는 0(비활성) 이상이어야 합니다: {top_k}")

        B, T      = input_ids.shape
        cur_attn  = attention_mask if attention_mask is not None else torch.ones(
            B, T, device=input_ids.device, dtype=torch.long)

        inputs_embeds, position_ids, next_pos = self._prepare_inputs(
            input_ids, pixel_values, image_grid_thw, cur_attn
        )
        # ★ 하이브리드 캐시는 프리필 '전에' 만들어 넘긴다 — linear attention 레이어가
        #   프리필 중에 conv/recurrent 상태를 저장할 자리가 필요하다(qwen3vl 과 다른 점).
        kv = self.model.language_model.new_cache()
        hidden, kv = self.model.language_model(
            inputs_embeds, attention_mask=cur_attn, position_ids=position_ids, kv_cache=kv,
        )
        # 배치 안전: 오른쪽 패딩이면 짧은 샘플의 마지막 위치는 pad 이므로,
        # 샘플별 실제 마지막 토큰 위치의 hidden 에서 첫 로짓을 뽑는다(B=1 무패딩이면 기존과 동일).
        last_idx    = cur_attn.sum(dim=-1) - 1                                  # (B,) 샘플별 마지막 실토큰 위치
        next_logits = self.lm_head(hidden[torch.arange(B, device=hidden.device), last_idx])   # (B, V)
        generated   = input_ids
        # 종료 토큰은 정수 하나 또는 여러 개(Qwen 은 <|im_end|>/<|endoftext|> 둘 다)일 수 있다.
        eos_ids  = ([eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)) if eos_token_id is not None else []
        eos_t    = torch.tensor(eos_ids, device=input_ids.device) if eos_ids else None
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)    # 샘플별 EOS 도달 여부

        for step in range(max_new_tokens):
            if do_sample:
                logits = next_logits / temperature
                if top_k and top_k > 0:
                    k   = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs      = torch.softmax(logits, dim=-1)
                next_token = _sample_top_p(probs, top_p)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # 이미 끝난 샘플은 EOS 를 강제해 뒤쪽에 쓰레기 토큰이 남지 않게 한다.
            if eos_t is not None and finished.any():
                next_token = torch.where(finished.unsqueeze(1), eos_t[0].expand_as(next_token), next_token)

            generated = torch.cat([generated, next_token], dim=-1)
            cur_attn  = torch.cat([cur_attn, torch.ones_like(next_token)], dim=-1)
            if eos_t is not None:
                finished = finished | (next_token == eos_t.view(1, -1)).any(dim=-1)
                if bool(finished.all()):                       # 샘플별 종료 추적(같은 스텝 요구 X)
                    break
            if step == max_new_tokens - 1:
                break

            emb          = self.get_input_embeddings()(next_token)              # (B, 1, D)
            pos          = (next_pos + step).view(1, B, 1).expand(3, -1, -1)    # (3, B, 1) 세 축 동시 증가
            hidden, kv   = self.model.language_model(
                emb, attention_mask=cur_attn, position_ids=pos, kv_cache=kv
            )                                                                   # (B, 1, D)
            next_logits  = self.lm_head(hidden[:, -1, :])                       # (B, V)

        return generated


# ------------------------------------------------------------------ #
# nucleus(top-p) 샘플링: 누적확률 p 초과 꼬리를 잘라 재정규화 후 multinomial.
# ------------------------------------------------------------------ #
def _sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum             = torch.cumsum(probs_sort, dim=-1)
    mask                  = (probs_sum - probs_sort) > p
    probs_sort[mask]      = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token            = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)
