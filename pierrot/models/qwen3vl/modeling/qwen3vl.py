"""Qwen3-VL 최상위 모델 (스크래치, 추론 전용).

구성: 동적 해상도 비전 타워(visual) + Qwen3 언어 디코더(language_model) + lm_head.
모듈 이름은 HF Qwen3-VL 체크포인트 키와 일치시켜 공개 가중치가
load_state_dict(strict=False) 로 바로 들어온다:
    model.visual.* / model.language_model.* / lm_head.weight(=임베딩 tie)

이미지 병합: 각 이미지는 비전 타워+머저를 거쳐 (t·h·w/m²) 개의 임베딩이 되고,
텍스트의 <|image_pad|> placeholder 자리에 순서대로 끼워 넣는다(masked_scatter).
시퀀스 길이는 보존된다 — 프로세서가 placeholder 개수를 정확히 그만큼 넣어 두었다.

M-RoPE: 위치가 (시간, 높이, 너비) 3축이다. 텍스트 구간은 세 축이 같은 값으로
1씩 증가하고, 이미지 구간은 격자 좌표를 따른다. 이미지가 끝나면 다음 텍스트는
max(h, w)/m 만큼 진행한 위치에서 이어진다(=이미지가 차지한 "위치 폭").

표준 causal LM 생성이다 — forward 는 로짓만 내고, generate 가 KV 캐시로 autoregressive
하게 이어 붙인다. 손실·활성화 체크포인팅 등 학습 경로는 학습 저장소 Pierrot-VLM-Lab 쪽에 있다.

텐서 차원 표기:
    B = 배치, T = 시퀀스 길이, D = text hidden, V = vocab
    S = 총 패치 수, m = spatial_merge_size, S/m² = 총 이미지 토큰 수
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Qwen3VLConfig
from .text import Qwen3VLTextModel
from .vision import Qwen3VLVisionModel


class Qwen3VLModel(nn.Module):
    """비전 타워 + 언어 디코더 묶음 (HF 키 model.* 에 대응)."""

    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config         = config
        self.visual         = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)


class Qwen3VLForConditionalGeneration(nn.Module):
    """Qwen3-VL: 동적 해상도 ViT + DeepStack + Qwen3 디코더 결합 VLM."""

    # ------------------------------------------------------------------ #
    # model(비전/언어)과 lm_head 를 조립한다. 이름은 HF 키와 일치.
    # ------------------------------------------------------------------ #
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config            = config
        self.model             = Qwen3VLModel(config)
        self.lm_head           = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.image_token_id    = config.image_token_id
        self.merge_size        = config.vision_config.spatial_merge_size

    # ------------------------------------------------------------------ #
    # lm_head 가중치를 텍스트 임베딩과 공유(weight tying).
    # Qwen3-VL-2B 는 tie 가 기본이라 체크포인트에 lm_head.weight 가 없다.
    # ------------------------------------------------------------------ #
    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    # ------------------------------------------------------------------ #
    # 텍스트 임베딩 테이블(이미지 병합·tie 기준).
    # ------------------------------------------------------------------ #
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.language_model.embed_tokens

    # ------------------------------------------------------------------ #
    # 패킹된 패치 시퀀스를 인코딩한다.
    #   pixel_values (S, patch_dim), image_grid_thw (n_images, 3)
    #   → (merged (S/m², D), deepstack [(S/m², D)] × 3)
    # ------------------------------------------------------------------ #
    def _encode_images(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor,
                       dtype: torch.dtype) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.model.visual(pixel_values.to(dtype), image_grid_thw)

    # ------------------------------------------------------------------ #
    # 텍스트 임베딩의 <|image_pad|> 자리를 이미지 임베딩으로 교체한다.
    # masked_scatter 는 마스크를 (배치, 시퀀스) 행 우선으로 순회하며 채우므로,
    # collate 가 배치 순서대로 쌓아 둔 이미지 패치 순서와 정확히 맞는다.
    # 개수가 다르면 프로세서/모델 설정 불일치이므로 즉시 오류를 낸다.
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
    #
    # 시퀀스를 텍스트 구간 / 이미지 구간으로 나눠 훑는다:
    #   - 텍스트 L 토큰 : 세 축 모두 pos, pos+1, ..., pos+L-1  → pos += L
    #   - 이미지 격자   : t/h/w 좌표를 meshgrid 로 펼쳐 pos 를 더한다.
    #                     차지하는 "위치 폭"은 max(h, w)/m (정사각이 아니어도 겹치지 않게)
    # 패딩 위치는 계산에서 제외하고 0 으로 둔다(어텐션 마스크가 어차피 가린다).
    # 반환하는 next_pos 는 생성(decode) 단계에서 이어 쓸 시작 위치다.
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
    # 반환: (inputs_embeds, position_ids, next_pos, visual_pos_masks, deepstack)
    # ------------------------------------------------------------------ #
    def _prepare_inputs(self, input_ids, pixel_values, image_grid_thw, attention_mask):
        inputs_embeds = self.get_input_embeddings()(input_ids)                 # (B, T, D)
        visual_masks, deepstack = None, None

        if pixel_values is not None:
            image_embeds, deepstack = self._encode_images(
                pixel_values, image_grid_thw, inputs_embeds.dtype
            )
            inputs_embeds, visual_masks = self._merge_image_features(
                input_ids, inputs_embeds, image_embeds
            )

        position_ids, next_pos = self.get_rope_index(input_ids, image_grid_thw, attention_mask)
        return inputs_embeds, position_ids, next_pos, visual_masks, deepstack

    # ------------------------------------------------------------------ #
    # 순전파. 이미지+텍스트를 받아 로짓을 낸다.
    #   ① 텍스트 임베딩 → ② 비전 인코딩(+DeepStack) → ③ <|image_pad|> 자리에 병합
    #   → ④ M-RoPE position_ids → ⑤ 언어 디코더(앞쪽 레이어에 DeepStack 주입) → hidden.
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
        inputs_embeds, position_ids, _, visual_masks, deepstack = self._prepare_inputs(
            input_ids, pixel_values, image_grid_thw, attention_mask
        )
        hidden, _ = self.model.language_model(
            inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
            visual_pos_masks=visual_masks, deepstack_visual_embeds=deepstack,
        )                                                                      # (B, T, D)

        if logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:, :]
        return {"logits": self.lm_head(hidden)}

    # ------------------------------------------------------------------ #
    # KV 캐시 기반 autoregressive 생성 (배치=1, 패딩 없는 프롬프트 가정).
    #   - 프리필: 이미지 병합된 프롬프트를 한 번에 처리(DeepStack 주입 포함)해
    #     캐시를 채우고 첫 토큰을 얻는다.
    #   - 디코드: 새 토큰 1개만 임베딩해 캐시 전체를 참조. 위치는 프리필이 돌려준
    #     next_pos 부터 세 축 모두 1씩 증가한다(이미지 뒤 텍스트와 동일 규칙).
    #     새 토큰은 이미지가 아니므로 DeepStack 주입은 하지 않는다.
    # do_sample 이면 top-k→top-p 샘플링(공식 generation_config: top_k=20, top_p=0.8),
    # 아니면 greedy. <eos> 만나면 종료. 전체 시퀀스 반환.
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
        # 추론 경계값 검증(학습엔 영향 없음; 잘못된 값이 조용히 이상 출력을 내는 것을 막는다).
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

        inputs_embeds, position_ids, next_pos, visual_masks, deepstack = self._prepare_inputs(
            input_ids, pixel_values, image_grid_thw, cur_attn
        )
        hidden, kv = self.model.language_model(
            inputs_embeds, attention_mask=cur_attn, position_ids=position_ids,
            visual_pos_masks=visual_masks, deepstack_visual_embeds=deepstack,
        )
        # 배치 안전: 오른쪽 패딩이면 짧은 샘플의 마지막 위치는 pad 이므로,
        # 샘플별 실제 마지막 토큰 위치의 hidden 에서 첫 로짓을 뽑는다(B=1 무패딩이면 기존과 동일).
        last_idx    = cur_attn.sum(dim=-1) - 1                                  # (B,)
        next_logits = self.lm_head(hidden[torch.arange(B, device=hidden.device), last_idx])
        generated   = input_ids
        # 종료 토큰은 정수 하나 또는 여러 개(Qwen 은 <|im_end|>/<|endoftext|> 둘 다)일 수 있다.
        eos_ids  = ([eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)) if eos_token_id is not None else []
        eos_t    = torch.tensor(eos_ids, device=input_ids.device) if eos_ids else None
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)    # 샘플별 EOS 도달 여부

        for step in range(max_new_tokens):
            if do_sample:
                logits = next_logits / temperature
                if top_k and top_k > 0:                    # top-k 필터(공식 recipe): k번째 미만 로짓 제거
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
            pos          = (next_pos + step).view(1, B, 1).expand(3, -1, -1)
            hidden, kv   = self.model.language_model(
                emb, attention_mask=cur_attn, position_ids=pos, kv_cache=kv
            )
            next_logits  = self.lm_head(hidden[:, -1, :])

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
