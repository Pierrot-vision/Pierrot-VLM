"""PaliGemma2 최상위 모델 (스크래치, 추론 전용).

구성: SigLIP 비전 타워 + 멀티모달 프로젝터 + Gemma2 언어 모델.

이 배포본이 유지하는 핵심 구조:
  - 패딩을 지원하는 prefix-LM 4D 어텐션 마스크.
    · 이미지 + 프리픽스(프롬프트) 토큰: 서로 양방향(bidirectional) 어텐션
    · 생성 토큰: causal 어텐션 (+ 프리픽스 전체를 볼 수 있음)
    · token_type_ids(0=prefix, 1=suffix)로 구분한다. 추론에서는 전부 prefix 다.
모듈 이름은 HF 체크포인트 키와 일치(vision_tower / multi_modal_projector / language_model).

손실 계산·활성화 체크포인팅 등 학습 경로는 학습 저장소 Pierrot-VLM-Lab 쪽에 있다.

텐서 차원 표기:
    B = 배치, L = 시퀀스 길이(이미지토큰 4096 + bos + prefix),
    H = hidden_size(2304), V = vocab(257216)
    입력: input_ids(B,L) / pixel_values(B,3,896,896) / attention_mask(B,L) /
          token_type_ids(B,L)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..config import PaliGemma2Config
from .gemma2 import Gemma2ForCausalLM, KVCache
from .projector import MultiModalProjector
from .siglip import SiglipVisionModel


class PaliGemma2ForConditionalGeneration(nn.Module):

    # ------------------------------------------------------------------ #
    # 세 구성요소를 조립한다: vision_tower(SigLIP) + multi_modal_projector(선형)
    # + language_model(Gemma2). 모듈 이름은 HF 체크포인트 키와 일치시켜
    # 공개 가중치가 load_state_dict 로 바로 들어오게 한다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: PaliGemma2Config):
        super().__init__()
        self.config                = config
        self.vision_tower          = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config)
        self.language_model        = Gemma2ForCausalLM(config.text_config)
        self.vocab_size            = config.vocab_size
        self.pad_token_id          = config.pad_token_id if config.pad_token_id is not None else -1

    # ------------------------------------------------------------------ #
    # lm_head 가중치를 embed_tokens 와 공유(weight tying)한다.
    # 실제 처리는 언어 모델로 위임한다.
    # ------------------------------------------------------------------ #
    def tie_weights(self) -> None:
        self.language_model.tie_weights()

    # ------------------------------------------------------------------ #
    # 텍스트 토큰 임베딩 테이블(embed_tokens)을 반환한다.
    # input_ids → 임베딩 변환과, <image> 자리 병합의 기준으로 쓰인다.
    # ------------------------------------------------------------------ #
    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    # ------------------------------------------------------------------ #
    # <image> placeholder 위치의 텍스트 임베딩을 프로젝션된 이미지 특징으로 교체.
    # 이미지 특징을 미리 /sqrt(hidden) 스케일해 두면, Gemma2 가 임베딩 전체에
    # 거는 ×sqrt(hidden) 정규화와 상쇄되어 이미지 토큰만 원 스케일로 유지된다.
    # 시퀀스 길이가 다르므로 where 가 아니라 masked_scatter 로 채운다.
    # ------------------------------------------------------------------ #
    def _merge_image_features(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        # 언어 hidden 을 직접 참조(최상위 config.hidden_size 오배선 방지). Gemma2 의
        # ×sqrt(hidden) 정규화와 정확히 상쇄되어야 하므로 text_config.hidden_size 를 쓴다.
        # 언어 모델 임베딩 차원
        hidden = self.config.text_config.hidden_size

        # Gemma2 의 ×sqrt(hidden) 과 상쇄되게 미리 나눔
        scaled = image_features / (hidden ** 0.5)
        # 텍스트 임베딩과 dtype 맞춤
        scaled = scaled.to(inputs_embeds.dtype)

        # <image> 위치만 True (B, L, 1)
        image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        # 임베딩 차원까지 확장 (B, L, H)
        image_mask = image_mask.expand_as(inputs_embeds)
        # True 자리에 이미지 특징을 순서대로 채워 넣음
        return inputs_embeds.masked_scatter(image_mask, scaled)

    # ------------------------------------------------------------------ #
    # 학습/프리필용 prefix-LM 4D additive 마스크 (B, 1, L, L) 를 만든다.
    # 허용 규칙: query i 가 key j 를 보려면
    #   j 가 유효 토큰  AND  ( j 가 prefix  OR  j <= i (causal) ).
    #   - 이미지+prefix 끼리: 서로 완전 양방향
    #   - suffix: 이전 토큰만(causal), 단 prefix 전체는 열람 가능
    #   - pad 키: 차단 / pad 쿼리 행: softmax NaN 방지로 대각선(self)만 허용
    # 허용=0, 차단=min_dtype 값으로 채운 additive 마스크를 반환한다.
    # ------------------------------------------------------------------ #
    def _build_prefixlm_mask(
        self,
        attention_mask: torch.Tensor,     # (B, L) 1=valid, 0=pad
        token_type_ids: torch.Tensor,     # (B, L) 0=prefix, 1=suffix
        dtype: torch.dtype,
    ) -> torch.Tensor:
        b, l      = attention_mask.shape
        device    = attention_mask.device
        min_value = torch.finfo(dtype).min

        causal        = torch.tril(torch.ones((l, l), dtype=torch.bool, device=device))   # (L, L) j<=i
        is_prefix_key = (token_type_ids == 0)                                           # (B, L)
        valid_key     = attention_mask.bool()                                           # (B, L)

        allowed = causal[None, :, :] | is_prefix_key[:, None, :]                    # (B, L, L)
        allowed = allowed & valid_key[:, None, :]

        # pad 쿼리 행이 전부 -inf 가 되어 softmax NaN 이 나는 것을 막기 위해 대각선 보장.
        eye     = torch.eye(l, dtype=torch.bool, device=device)[None, :, :]
        allowed = allowed | eye

        mask = torch.where(allowed, 0.0, min_value).to(dtype)
        return mask.unsqueeze(1)                                                    # (B, 1, L, L)

    # ------------------------------------------------------------------ #
    # RoPE 용 position_ids 를 만든다.
    # 유효 토큰을 앞에서부터 0,1,2,... 로 누적 카운트(cumsum-1)하고,
    # pad 위치는 1 로 채워 회전 인덱스가 음수가 되지 않게 한다.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    # ------------------------------------------------------------------ #
    # 순전파. 이미지+텍스트를 받아 로짓을 낸다.
    # 흐름: ① 텍스트 임베딩 → ② 비전 인코딩·프로젝션 → ③ <image> 자리에 병합
    #       → ④ prefix-LM 마스크/위치 id → ⑤ Gemma2 몸통 → hidden → lm_head.
    # logits_to_keep>0 이면 마지막 N 위치의 로짓만 계산해, 896에서 전체 (B,L,vocab)
    # FP32 로짓(~16GiB)을 만들지 않는다(0=전체, 하위호환).
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # ① 텍스트: input_ids(B,L) → 임베딩 (B, L, H). <image> 자리는 임시값(아래서 교체)
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # ② 이미지: pixel_values(B,3,896,896) → SigLIP 패치특징 (B, N=4096, 1152)
        image_outputs  = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # ③ 프로젝터: (B, N, 1152) → (B, N, H) 언어 임베딩 차원으로 투영
        image_features = self.multi_modal_projector(image_outputs)
        # ④ <image> 자리(N개)에 이미지 특징 삽입 → (B, L, H)
        inputs_embeds  = self._merge_image_features(input_ids, inputs_embeds, image_features)

        # 마스크: token_type_ids 가 없으면(=전부 prefix) 완전 양방향 취급. (B, L)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # prefix-LM 4D additive 어텐션 마스크 (B, 1, L, L)
        causal_mask  = self._build_prefixlm_mask(attention_mask, token_type_ids, inputs_embeds.dtype)
        # RoPE 용 위치 id (B, L)
        position_ids = self._position_ids(attention_mask)

        # ⑤ 언어 모델 '몸통'만 통과시켜 hidden 을 얻는다(lm_head 는 아래에서 선택 적용).
        hidden = self.language_model.model(causal_mask, position_ids, inputs_embeds, kv_cache=None)  # (B, L, H)

        # 필요한 만큼만 projection (기본 0=전체). 896에선 마지막 위치만 권장.
        if logits_to_keep > 0:
            # 마지막 N 위치만 남김
            hidden = hidden[:, -logits_to_keep:, :]
        # hidden → vocab 로짓
        return {"logits": self.language_model.compute_logits(hidden)}

    # ------------------------------------------------------------------ #
    # KV 캐시 기반 autoregressive 생성. (배치=1, 패딩 없는 프롬프트 가정)
    #   - step 0 (프리필): 이미지 병합된 프롬프트를 양방향으로 한 번에 처리해
    #                      각 레이어의 K/V 를 캐시에 채우고 첫 토큰을 뽑는다.
    #   - step≥1 (디코드): 새 토큰 1개만 임베딩해 캐시 전체를 참조(마스크 0).
    # do_sample 이면 top-p 샘플링, 아니면 greedy(argmax). <eos> 만나면 종료.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        return_scores: bool = False,
        presence_token_ids: Optional[torch.Tensor] = None,
        presence_margin: Optional[float] = None,
        presence_loc_mass: Optional[float] = None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        if presence_token_ids is not None and not return_scores:
            raise ValueError("presence_token_ids 는 return_scores=True 와 함께 사용해야 합니다.")
        if (presence_margin is not None or presence_loc_mass is not None) and presence_token_ids is None:
            raise ValueError("presence 강제(margin/loc_mass)는 presence_token_ids 가 필요합니다.")
        if (presence_margin is not None or presence_loc_mass is not None) and do_sample:
            raise ValueError("presence 강제(margin/loc_mass)는 greedy 디코딩에서만 쓴다.")
        if presence_token_ids is not None and eos_token_id is None:
            raise ValueError("presence 신호 계산에는 eos_token_id 가 필요합니다.")
        # 레이어별 K/V 를 담아둘 캐시
        kv_cache  = KVCache()
        # return_scores 일 때 각 생성 토큰의 선택 확률을 모아둔다(검출 신뢰도용)
        token_probs = []
        presence_signals: Dict[str, torch.Tensor] = {}
        presence_forced = None                              # step 0 에서 EOS 를 뚫고 강제됐는지
        # 지금까지 생성된 전체 시퀀스
        generated = input_ids
        # 시퀀스에 맞춰 늘려갈 attention_mask
        cur_attn  = attention_mask

        # 프리필: 프롬프트(이미지+프리픽스)는 전부 양방향 -> token_type 전부 0
        # ① 텍스트: 토큰 id → 임베딩
        inputs_embeds  = self.get_input_embeddings()(input_ids)
        # ② 이미지 패치 → 특징 인코딩
        image_outputs  = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # ③ 언어 임베딩 차원으로 투영
        image_features = self.multi_modal_projector(image_outputs)
        # ④ <image> 자리에 이미지 특징 삽입
        inputs_embeds  = self._merge_image_features(input_ids, inputs_embeds, image_features)

        for step in range(max_new_tokens):
            if step == 0:
                # 프리필: 프롬프트 전체를 한 번에 처리 (전부 prefix)
                token_type_ids = torch.zeros_like(generated)
                # prefix-LM 마스크 (프롬프트 내부는 양방향)
                mask           = self._build_prefixlm_mask(cur_attn, token_type_ids, inputs_embeds.dtype)
                # RoPE 위치 id
                position_ids   = self._position_ids(cur_attn)
                # 이미 병합해 둔 프롬프트 임베딩 사용
                embeds         = inputs_embeds
            else:
                # 디코드: 새 토큰 1개만 임베딩
                embeds       = self.get_input_embeddings()(next_token)
                # 이번에 넣는 쿼리 길이 = 1
                q_len        = 1
                # 캐시된 토큰 + 새 토큰 = 참조할 전체 길이
                kv_len       = kv_cache.num_items() + q_len
                # 캐시 전체를 볼 수 있게 마스크 0 (차단 없음)
                mask         = torch.zeros((1, 1, q_len, kv_len), dtype=embeds.dtype, device=embeds.device)
                # 새 토큰의 위치 id (마지막 자리)
                position_ids = cur_attn.long().cumsum(-1)[:, -1:] - 1

            # 몸통에서 hidden 만 받고 '마지막 위치'만 lm_head 에 통과시킨다.
            # 프리필(embeds 길이 L~4096)에서 전체 (B,L,vocab) FP32 로짓을 만들지 않아
            # 추론 메모리/속도를 크게 아낀다(다음 토큰은 마지막 위치 로짓만 필요).
            # 언어 모델 몸통 통과 → hidden
            hidden      = self.language_model.model(mask, position_ids, embeds, kv_cache)
            # 마지막 위치 hidden 만 vocab 로짓으로
            next_logits = self.language_model.compute_logits(hidden[:, -1:, :])[:, -1, :]

            # 점수 기록·presence 강제에 원본 모델 분포가 필요하다.
            need_probs  = return_scores or presence_margin is not None
            model_probs = torch.softmax(next_logits, dim=-1) if need_probs else None

            if do_sample:
                # 생성 정책용 분포: 온도 적용 후 top-p 샘플링.
                sampling_probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token     = _sample_top_p(sampling_probs, top_p)
            else:
                # greedy: 가장 확률 높은 토큰
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # presence 강제(greedy 전용). 박스 시작 좌표가 여러 <loc> bin 에 흩어지면
            # 개별 최댓값이 EOS 를 못 넘어 greedy 가 EOS 를 골라 객체를 통째로 놓친다.
            # 하지만 <loc> 확률 '합'(loc_mass)이 크면 모델은 "박스가 있다"고 확신하는 것
            # → EOS 대신 최댓값 <loc> 토큰을 강제한다. 두 기준을 지원한다(OR 결합):
            #   - presence_margin  : (loc_mass - eos) > margin    (상대 기준)
            #   - presence_loc_mass: loc_mass >= 이 값            (절대 기준, 예: 0.7)
            # 절대 기준은 "loc_mass 가 0.7 이상이면 박스가 있다"처럼 직관적으로 쓴다.
            if presence_margin is not None or presence_loc_mass is not None:
                loc_ids   = presence_token_ids.to(device=model_probs.device, dtype=torch.long)
                loc_probs = model_probs.index_select(-1, loc_ids)      # (B, 1024)
                loc_mass  = loc_probs.sum(dim=-1)                      # (B,)
                eos_prob  = model_probs[:, eos_token_id]               # (B,)
                chose_eos = (next_token.squeeze(-1) == eos_token_id)
                cond      = torch.zeros_like(chose_eos)
                if presence_margin is not None:
                    cond = cond | ((loc_mass - eos_prob) > presence_margin)
                if presence_loc_mass is not None:
                    cond = cond | (loc_mass >= presence_loc_mass)
                force     = chose_eos & cond
                if force.any():
                    best_loc = loc_ids[loc_probs.argmax(dim=-1)]       # (B,) 최댓값 loc 토큰
                    next_token = torch.where(
                        force.unsqueeze(-1), best_loc.unsqueeze(-1), next_token
                    )
                # step 0 에서 강제가 일어났으면 "이 생성은 복원된 것"으로 표시해 둔다.
                # (로그에서 어느 검출이 EOS 를 뚫고 살아났는지 추적 가능하게)
                if step == 0:
                    presence_forced = force

            # 검출 confidence 는 생성 정책과 분리한다. temperature/top-p 적용 전 원본
            # 모델 분포에서 선택 토큰의 확률을 기록해야 설정이 달라도 점수를 비교할 수 있다.
            if return_scores:
                token_probs.append(model_probs.gather(-1, next_token))
                # per-class 현재 체크포인트의 존재 신호를 진단한다. 첫 생성 위치에서
                # 유효한 모든 <loc> 토큰 확률 질량과 EOS 확률을 별도로 보존한다.
                if step == 0 and presence_token_ids is not None:
                    loc_ids = presence_token_ids.to(device=model_probs.device, dtype=torch.long)
                    presence_signals = {
                        "loc_mass": model_probs.index_select(-1, loc_ids).sum(dim=-1),
                        "eos_prob": model_probs[:, eos_token_id],
                    }
                    # presence 강제로 EOS 를 뚫고 살아난 생성이면 표시(로그 추적용)
                    if presence_forced is not None:
                        presence_signals["forced"] = presence_forced

            # 생성 결과에 새 토큰 이어붙임
            generated = torch.cat([generated, next_token], dim=-1)
            # 새 토큰 자리를 valid(1)로 마스크에 추가
            cur_attn  = torch.cat([cur_attn, torch.ones_like(next_token)], dim=-1)

            # <eos> 나오면 종료
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        if return_scores:
            # (B, 생성길이) 각 생성 토큰이 선택될 확률
            scores = torch.cat(token_probs, dim=-1) if token_probs else torch.empty(
                input_ids.shape[0], 0, dtype=torch.float32, device=input_ids.device
            )
            if presence_token_ids is not None:
                return generated, scores, presence_signals
            return generated, scores
        return generated


# ------------------------------------------------------------------ #
# nucleus(top-p) 샘플링.
# 확률을 내림차순 정렬해 누적확률이 p 를 넘는 꼬리를 0 으로 잘라내고,
# 재정규화한 뒤 multinomial 로 하나 뽑아 원래 vocab 인덱스로 되돌린다.
# ------------------------------------------------------------------ #
def _sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum             = torch.cumsum(probs_sort, dim=-1)
    mask                  = (probs_sum - probs_sort) > p
    probs_sort[mask]      = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token            = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)
