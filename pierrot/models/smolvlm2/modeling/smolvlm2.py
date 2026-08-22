"""SmolVLM2 최상위 모델 (스크래치, 추론 전용).

구성: SigLIP 비전 인코더(vision_model) + 픽셀셔플 커넥터(connector) +
SmolLM2 언어 디코더(text_model) + lm_head. 모듈 이름은 HF SmolVLM(Idefics3)
체크포인트 키와 일치시켜 공개 가중치가 load_state_dict(strict=False) 로 바로 들어온다:
    model.vision_model.* / model.connector.* / model.text_model.* / lm_head.weight

이미지 병합(inputs_merger): 각 이미지는 커넥터를 거쳐 S(=image_seq_len)개의 임베딩이
되고, 텍스트의 <image> placeholder 자리(S개 단위)에 순서대로 끼워 넣는다. 시퀀스
길이는 보존된다(<image> 토큰 1개 ↔ 이미지 임베딩 1행 교체).

표준 causal LM 생성이다 — forward 는 로짓만 내고, generate 가 KV 캐시로 autoregressive
하게 이어 붙인다. 손실·활성화 체크포인팅 등 학습 경로는 학습 저장소 Pierrot-VLM-Lab 쪽에 있다.

텐서 차원 표기:
    B = 배치, T = 시퀀스 길이, D = text hidden, V = vocab
    num_images = 배치×샘플의 최대 타일 수, N_real = 패딩 제외 실제 타일 수, S = image_seq_len
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..config import SmolVLM2Config
from .connector import SmolVLMConnector
from .text import SmolLM2TextModel
from .vision import SmolVLMVisionTransformer


class SmolVLMModel(nn.Module):
    """비전 + 커넥터 + 텍스트 디코더 묶음 (HF 키 model.* 에 대응)."""

    def __init__(self, config: SmolVLM2Config):
        super().__init__()
        self.config       = config
        self.vision_model = SmolVLMVisionTransformer(config.vision_config)
        self.connector    = SmolVLMConnector(config)
        self.text_model   = SmolLM2TextModel(config.text_config)


class SmolVLM2ForConditionalGeneration(nn.Module):
    """SmolVLM2: SigLIP + 픽셀셔플 커넥터 + SmolLM2 디코더 결합 VLM."""

    # ------------------------------------------------------------------ #
    # model(비전/커넥터/텍스트)과 lm_head 를 조립한다. 이름은 HF 키와 일치.
    # ------------------------------------------------------------------ #
    def __init__(self, config: SmolVLM2Config):
        super().__init__()
        self.config         = config
        self.model          = SmolVLMModel(config)
        self.lm_head        = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.image_token_id = config.image_token_id
        self.image_seq_len  = config.image_seq_len

    # ------------------------------------------------------------------ #
    # lm_head 가중치를 텍스트 임베딩과 공유(weight tying).
    # SmolLM2 는 tie 가 기본이며, 공개 체크포인트가 lm_head 를 따로 담지 않아도
    # 임베딩 공유로 채워진다.
    # ------------------------------------------------------------------ #
    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.text_model.embed_tokens.weight

    # ------------------------------------------------------------------ #
    # 텍스트 임베딩 테이블(이미지 병합·tie 기준).
    # ------------------------------------------------------------------ #
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.text_model.embed_tokens

    # ------------------------------------------------------------------ #
    # RoPE/causal 용 position_ids: 유효 토큰을 앞에서부터 0,1,2,... 로 센다.
    # pad 위치는 1 로 채워 음수 인덱스를 막는다(마스크로 어차피 제외됨).
    # ------------------------------------------------------------------ #
    @staticmethod
    def _position_ids(attention_mask: Optional[torch.Tensor], B: int, T: int, device) -> torch.Tensor:
        if attention_mask is None:
            return torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    # ------------------------------------------------------------------ #
    # pixel_values(B, num_images, C, H, W) → 실제 타일만 골라 비전+커넥터 인코딩.
    # 배치 정렬용 패딩 타일(전부 0)은 버린다(HF Idefics3 규약). 실제 타일이 하나도
    # 없으면 그래프 유지를 위해 첫 타일 하나를 남긴다.
    # 반환: (N_real, S, D) 커넥터 출력.
    # ------------------------------------------------------------------ #
    def _encode_images(self, pixel_values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        B, num_images = pixel_values.shape[:2]
        pv = pixel_values.to(dtype).view(B * num_images, *pixel_values.shape[2:])

        nb_per_image = pv.shape[1:].numel()
        real         = (pv == 0.0).sum(dim=(-1, -2, -3)) != nb_per_image
        if not real.any():
            real[0] = True
        pv = pv[real].contiguous()

        image_hidden = self.model.vision_model(pv)            # (N_real, N_patches, Dv)
        return self.model.connector(image_hidden)             # (N_real, S, D)

    # ------------------------------------------------------------------ #
    # 텍스트 임베딩의 <image> 자리를 이미지 임베딩으로 교체한다(out-of-place).
    #   - 샘플별 <image> 토큰 수는 S 의 배수(이미지 1장 = S 토큰)여야 한다.
    #   - image_hidden_states[offset] (S, D) 한 블록이 이미지 한 장에 대응하며,
    #     블록은 배치 순서대로 소비된다(collate 가 타일을 그 순서로 쌓는다).
    # 시퀀스 길이는 보존된다. (HF SmolVLMModel.inputs_merger 와 동일 로직)
    # ------------------------------------------------------------------ #
    def _merge_image_features(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        _, T, _ = inputs_embeds.shape
        _, S, _ = image_hidden_states.shape

        image_offset = 0
        merged: List[torch.Tensor] = []
        for cur_ids, cur_embeds in zip(input_ids, inputs_embeds):
            positions = (cur_ids == self.image_token_id).nonzero(as_tuple=True)[0]
            n_img_tok = positions.numel()

            if n_img_tok == 0:
                # 텍스트 전용: 이미지 블록을 소비하지 않되, 인코더가 그래프에 남도록
                # 길이 0 슬라이스를 이어붙인다(분산 학습 안정).
                merged.append(torch.cat([cur_embeds, image_hidden_states[0][:0, :]], dim=0))
                continue
            if n_img_tok % S != 0:
                raise ValueError(f"<image> 토큰 수({n_img_tok})가 S({S})의 배수가 아닙니다.")

            pos_list  = positions.tolist()
            segments  = []
            text_from = 0
            for c in range(0, n_img_tok, S):                  # 이미지 1장(S 토큰)씩
                block = image_hidden_states[image_offset]     # (S, D)
                image_offset += 1
                for i_s, pos in enumerate(pos_list[c:c + S]):
                    if pos > text_from:
                        segments.append(cur_embeds[text_from:pos])
                    segments.append(block[i_s:i_s + 1, :])
                    text_from = pos + 1
            if text_from < T:
                segments.append(cur_embeds[text_from:])
            merged.append(torch.cat(segments, dim=0))

        return torch.stack(merged)

    # ------------------------------------------------------------------ #
    # 순전파. 이미지+텍스트를 받아 로짓을 낸다.
    #   ① 텍스트 임베딩 → ② 비전 인코딩·커넥터 → ③ <image> 자리에 병합
    #   → ④ position_ids → ⑤ 텍스트 디코더 → hidden.
    # logits_to_keep>0 이면 마지막 N 위치의 로짓만 만든다(생성 프리필에서 (B,T,V)
    # 전체를 만들지 않아 메모리를 아낀다).
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        inputs_embeds = self.get_input_embeddings()(input_ids)                 # (B, T, D)

        if pixel_values is not None:
            image_hidden = self._encode_images(pixel_values, inputs_embeds.dtype)
            inputs_embeds = self._merge_image_features(input_ids, inputs_embeds, image_hidden)

        B, T, _      = inputs_embeds.shape
        position_ids = self._position_ids(attention_mask, B, T, inputs_embeds.device)
        hidden, _    = self.model.text_model(
            inputs_embeds, attention_mask=attention_mask, position_ids=position_ids
        )                                                                      # (B, T, D)

        if logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:, :]
        return {"logits": self.lm_head(hidden)}

    # ------------------------------------------------------------------ #
    # KV 캐시 기반 autoregressive 생성 (배치=1, 패딩 없는 프롬프트 가정).
    #   - 프리필: 이미지 병합된 프롬프트를 한 번에 처리해 캐시를 채우고 첫 토큰 획득
    #   - 디코드: 새 토큰 1개만 임베딩해 캐시 전체를 참조
    # do_sample 이면 top-p 샘플링, 아니면 greedy. <eos> 만나면 종료. 전체 시퀀스 반환.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_hidden  = self._encode_images(pixel_values, inputs_embeds.dtype)
            inputs_embeds = self._merge_image_features(input_ids, inputs_embeds, image_hidden)

        B, T, _   = inputs_embeds.shape
        generated = input_ids
        cur_attn  = attention_mask if attention_mask is not None else torch.ones(B, T, device=input_ids.device, dtype=torch.long)

        # 프리필: 프롬프트 전체 → 캐시 + 마지막 위치 로짓.
        position_ids  = self._position_ids(cur_attn, B, T, inputs_embeds.device)
        hidden, kv    = self.model.text_model(inputs_embeds, attention_mask=cur_attn, position_ids=position_ids)
        next_logits   = self.lm_head(hidden[:, -1, :])

        for step in range(max_new_tokens):
            if do_sample:
                probs      = torch.softmax(next_logits / temperature, dim=-1)
                next_token = _sample_top_p(probs, top_p)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            cur_attn  = torch.cat([cur_attn, torch.ones_like(next_token)], dim=-1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            if step == max_new_tokens - 1:
                break

            emb          = self.get_input_embeddings()(next_token)              # (B, 1, D)
            start_pos    = kv[0]["key"].size(2)                                  # 캐시 길이
            position_ids = torch.full((B, 1), start_pos, device=emb.device, dtype=torch.long)
            hidden, kv   = self.model.text_model(emb, attention_mask=cur_attn, position_ids=position_ids, kv_cache=kv)
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
