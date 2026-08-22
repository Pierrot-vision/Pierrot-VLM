"""Gemma2 언어 모델 (스크래치 구현).

Gemma1 대비 핵심 차이를 모두 반영한다:
  - 레이어마다 4개의 RMSNorm (input / post-attn / pre-ff / post-ff)
  - local(sliding window) / global attention 교차
  - attention logit soft-capping, final logit soft-capping
  - query_pre_attn_scalar 로 attention 스케일
  - GQA (num_key_value_heads < num_attention_heads)

모듈/파라미터 이름은 HF 체크포인트 키
(language_model.model.layers.N.self_attn.q_proj ...)와 일치시켰다.

이 모델은 추론(KV cache)뿐 아니라 학습도 지원한다. 학습 시에는 호출부에서
prefix-LM 4D additive 마스크(paligemma2.py 에서 생성)를 넘겨준다.

텐서 차원 표기:
    B = 배치, L=q_len = 시퀀스 길이, kv_len = 캐시 포함 key 길이,
    H = hidden_size(2304), D = head_dim(256), V = vocab(257216)
    num_heads(8) = Q 헤드 수, num_kv_heads(4) = KV 헤드 수(GQA)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Gemma2Config


class KVCache:
    """레이어별 K/V 를 시퀀스 축으로 이어 붙이는 단순 캐시 (추론용)."""

    # ------------------------------------------------------------------ #
    # 레이어별 K/V 를 담을 빈 리스트를 준비한다.
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor]   = []
        self.value_cache: List[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # 현재 캐시에 쌓인 시퀀스 길이를 반환한다(비었으면 0).
    # 디코드 시 position/마스크 길이 계산에 쓰인다.
    # ------------------------------------------------------------------ #
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        return self.key_cache[0].shape[-2]

    # ------------------------------------------------------------------ #
    # layer_idx 레이어의 K/V 를 갱신하고 누적된 전체 K/V 를 반환한다.
    # 첫 호출이면 새로 저장하고, 이후엔 시퀀스 축(dim=-2)으로 이어 붙인다.
    # ------------------------------------------------------------------ #
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class Gemma2RMSNorm(nn.Module):
    """weight 를 0 으로 초기화하고 (1 + weight) 로 스케일하는 Gemma 방식 RMSNorm."""

    # ------------------------------------------------------------------ #
    # 스케일 weight 를 0 으로 초기화한다(실효 게인은 1+weight 라 초기 1.0).
    # ------------------------------------------------------------------ #
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    # ------------------------------------------------------------------ #
    # RMS(제곱평균 제곱근)로 마지막 차원을 정규화한다.
    # ------------------------------------------------------------------ #
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # ------------------------------------------------------------------ #
    # float32 로 정규화 후 (1+weight) 게인을 곱하고 원래 dtype 으로 되돌린다.
    # float 승격은 bf16/fp16 에서의 수치 안정을 위한 것.
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:(..., H) 모양 불변. 마지막 차원(H)만 정규화한다.
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Gemma2RotaryEmbedding(nn.Module):
    """RoPE 회전 각도용 cos/sin 생성."""

    # ------------------------------------------------------------------ #
    # 역주파수 inv_freq = 1/base^(2i/dim) 를 버퍼로 등록한다(학습 안 함).
    # ------------------------------------------------------------------ #
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim  = dim
        self.base = base
        inv_freq  = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------ #
    # position_ids 로 각 위치의 회전각을 만들어 (cos, sin) 을 반환한다.
    # autocast 를 꺼 float32 로 계산해 정밀도를 지킨 뒤 입력 dtype 으로 캐스팅.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, L)
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos      = position_ids[:, None, :].float()

        # autocast 하에서도 float32 로 계산 (수치 안정)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq @ pos).transpose(1, 2)               # (B, L, dim/2)
            emb   = torch.cat((freqs, freqs), dim=-1)              # (B, L, dim)
            cos   = emb.cos()
            sin   = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


# ------------------------------------------------------------------ #
# RoPE 회전용: 뒤 절반 차원의 부호를 바꿔 앞으로 붙인다(90° 회전 성분).
# ------------------------------------------------------------------ #
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ------------------------------------------------------------------ #
# Q, K 에 회전 위치 임베딩(RoPE)을 적용한다.
# x*cos + rotate_half(x)*sin 로 위치 정보를 주파수 회전으로 주입한다.
# ------------------------------------------------------------------ #
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    # cos/sin: (B, L, D) → (B, 1, L, D) 로 헤드축 브로드캐스트
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # q,k: (B, heads, L, D) 모양 불변, 회전만 적용
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ------------------------------------------------------------------ #
# GQA 용 KV 헤드 확장: (B, num_kv, L, D) -> (B, num_kv*n_rep, L, D).
# KV 헤드를 n_rep 번 복제해 Q 헤드 수에 맞춘다.
# ------------------------------------------------------------------ #
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, num_kv, slen, hd = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, num_kv, n_rep, slen, hd)
    return hidden_states.reshape(b, num_kv * n_rep, slen, hd)


class Gemma2MLP(nn.Module):
    """GeGLU 게이트 MLP."""

    # ------------------------------------------------------------------ #
    # gate/up(확장)·down(축소) 선형층을 만든다(모두 bias 없음).
    # ------------------------------------------------------------------ #
    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    # ------------------------------------------------------------------ #
    # GeGLU: gelu(gate(x)) * up(x) 로 게이팅 후 down 으로 축소한다.
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:(B,L,H) → gate,up:(B,L,intermediate) → 게이팅 곱 → down:(B,L,H)
        gate = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


class Gemma2Attention(nn.Module):
    """GQA + RoPE + soft-capping + (local/global) 마스크 지원 어텐션."""

    # ------------------------------------------------------------------ #
    # Q/K/V/출력 투영(GQA: KV 헤드 수가 더 적음)과 RoPE 모듈을 만든다.
    # 스케일은 1/sqrt(query_pre_attn_scalar), 레이어별 local/global 유형과
    # soft-capping/sliding_window 값을 config 에서 읽어 둔다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.config        = config
        self.layer_idx     = layer_idx
        self.num_heads     = config.num_attention_heads
        self.num_kv_heads  = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.head_dim      = config.head_dim
        # Gemma2: scale = 1/sqrt(query_pre_attn_scalar)
        self.scaling                = config.query_pre_attn_scalar ** -0.5
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.sliding_window         = config.sliding_window
        self.attn_type              = config.attn_types[layer_idx]
        self.attention_dropout      = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.rotary_emb = Gemma2RotaryEmbedding(self.head_dim, base=config.rope_theta)

    # ------------------------------------------------------------------ #
    # local_sliding 레이어용: 슬라이딩 윈도우 '밴드' 밖을 추가 마스킹한다.
    # prefix-LM 은 prefix 를 양방향으로 보므로, local 레이어는 현재 위치 기준
    # 과거와 미래 '양쪽' 모두 |i - j| < window 안이어야 한다(=밴드 마스크).
    # 디코드 단계(kv_len != q_len)에선 쿼리 절대위치가 kv_len-q_len 만큼 밀려 있어
    # 그 offset 을 반영해 거리를 계산한다.
    # ------------------------------------------------------------------ #
    def _sliding_window_mask(self, attention_mask: torch.Tensor, q_len: int, kv_len: int) -> torch.Tensor:
        min_value = torch.finfo(attention_mask.dtype).min
        # (q_len, kv_len) 에서 |i - j| >= sliding_window 인 곳을 마스킹(과거/미래 양쪽)
        q_idx = torch.arange(q_len, device=attention_mask.device)
        # 디코드 단계(kv_len != q_len)에서는 쿼리 절대위치가 kv_len-q_len 만큼 밀려 있다.
        offset  = kv_len - q_len
        k_idx   = torch.arange(kv_len, device=attention_mask.device)
        dist    = (q_idx[:, None] + offset) - k_idx[None, :]
        outside = dist.abs() >= self.sliding_window                 # 윈도우 밴드 밖(과거·미래)
        outside = outside[None, None, :, :]
        return attention_mask.masked_fill(outside, min_value)

    # ------------------------------------------------------------------ #
    # 어텐션 순전파.
    # Q/K/V 투영 → RoPE → (추론 시) KV 캐시 갱신 → GQA 로 KV 확장 →
    # (local 이면 sliding 마스크 합성) → 어텐션. 마지막 단계는 두 갈래:
    #   · soft-capping OFF (Stage 3): scaled_dot_product_attention(SDPA)
    #     → head 별 L×L score 를 저장하지 않아 896(4096 토큰) 메모리에 필수.
    #     (단 4D additive 마스크를 주므로 최신 GPU 에선 Flash 가 아니라 memory-efficient
    #      backend 가 선택된다. 그래도 절감 효과는 크다.)
    #   · soft-capping ON (사전학습 재현): 수동 score→tanh soft-cap→mask→softmax.
    # SDPA 는 soft-capping 을 fuse 하지 못하므로 OFF 일 때만 사용한다.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # hidden_states: (B, q_len, H)
        b, q_len, _ = hidden_states.shape

        # Q: (B, num_heads, q_len, D) — 헤드 분할 후 (heads, seq) 축 전치
        q = self.q_proj(hidden_states).view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V: (B, num_kv_heads, q_len, D) — GQA 라 KV 헤드 수가 Q 보다 적음
        k = self.k_proj(hidden_states).view(b, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE 회전각(cos/sin)을 Q, K 에 적용해 위치정보 주입 (모양 불변)
        cos, sin = self.rotary_emb(v, position_ids)
        q, k     = apply_rotary_pos_emb(q, k, cos, sin)

        # 추론: 과거 K, V 를 캐시에 이어붙임 → (B, num_kv_heads, kv_len, D)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        # GQA: KV 헤드를 num_kv_groups 배 복제해 Q 헤드 수와 맞춤 → (B, num_heads, kv_len, D)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # prefix-LM 마스크에 (local 레이어면) sliding-window 밴드를 합성.
        mask = attention_mask
        if self.attn_type == "local_sliding" and self.sliding_window is not None:
            mask = self._sliding_window_mask(mask, q_len, k.shape[-2])

        if self.attn_logit_softcapping is None:
            # SDPA: head 별 score 를 통째로 만들지 않음(896 필수). 4D additive mask →
            # 최신 GPU 에선 memory-efficient backend 선택(Flash 아님)이지만 절감 효과 큼.
            dropout_p = self.attention_dropout if self.training else 0.0
            out = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p, scale=self.scaling,
            )
        else:
            # 수동 path: soft-capping 을 살리려면 score 를 명시적으로 만들어야 한다.
            attn = (q @ k.transpose(-2, -1)) * self.scaling        # (B, heads, q, kv)
            cap  = self.attn_logit_softcapping
            attn = torch.tanh(attn / cap) * cap
            attn = attn + mask                                     # additive 마스크(-inf/0)
            attn = attn.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
            attn = nn.functional.dropout(attn, p=self.attention_dropout, training=self.training)
            out  = attn @ v                                        # (B, heads, q, head_dim)

        # (B, num_heads, q_len, D) → (B, q_len, H) 헤드 병합 후 출력 투영
        out = out.transpose(1, 2).contiguous().view(b, q_len, -1)
        return self.o_proj(out)


class Gemma2DecoderLayer(nn.Module):
    """Gemma2 디코더 레이어 (4개의 RMSNorm sandwich)."""

    # ------------------------------------------------------------------ #
    # 어텐션/MLP 서브블록과, 각 서브블록을 앞뒤로 감싸는 RMSNorm 4개
    # (input / post-attention / pre-feedforward / post-feedforward)를 만든다.
    # 이 pre+post 샌드위치가 Gemma1 대비 Gemma2 의 핵심 차이다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.self_attn                  = Gemma2Attention(config, layer_idx)
        self.mlp                        = Gemma2MLP(config)
        self.input_layernorm            = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm   = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm  = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # ------------------------------------------------------------------ #
    # 두 잔차 블록을 통과시킨다.
    #   ① pre-norm → self_attn → post-norm → +residual
    #   ② pre-norm → mlp       → post-norm → +residual
    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # Attention 블록: pre-norm -> attn -> post-norm -> 잔차
        residual      = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, kv_cache)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Feedforward 블록: pre-norm -> mlp -> post-norm -> 잔차
        residual      = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Gemma2Model(nn.Module):
    """임베딩 + 디코더 스택 + 최종 norm."""

    # ------------------------------------------------------------------ #
    # 토큰 임베딩, 디코더 레이어 스택, 최종 RMSNorm 을 조립한다.
    # ------------------------------------------------------------------ #
    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.config       = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers       = nn.ModuleList(
            [Gemma2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm         = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # ------------------------------------------------------------------ #
    # 토큰 임베딩 테이블을 반환한다.
    # ------------------------------------------------------------------ #
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    # ------------------------------------------------------------------ #
    # 임베딩에 ×sqrt(hidden) 정규화를 건 뒤 디코더 스택을 통과시키고 최종 norm.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
    
        # inputs_embeds: (B, L, H). Gemma 는 임베딩에 ×sqrt(H) 정규화를 건다.
        # (paligemma2 의 이미지특징 /sqrt(H) 스케일이 이걸 상쇄 → 이미지토큰만 원 스케일)
        normalizer    = torch.tensor(self.config.hidden_size ** 0.5, dtype=inputs_embeds.dtype)
        hidden_states = inputs_embeds * normalizer

        # 26개 디코더 레이어 통과(모양 불변 (B, L, H))
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids, kv_cache)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma2ForCausalLM(nn.Module):
    """Gemma2Model + lm_head (임베딩과 weight tying)."""

    # ------------------------------------------------------------------ #
    # 몸통(Gemma2Model)과 출력 헤드(lm_head)를 만든다. lm_head 는 bias 없음.
    # ------------------------------------------------------------------ #
    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.config     = config
        self.model      = Gemma2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head    = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # ------------------------------------------------------------------ #
    # 토큰 임베딩 테이블(몸통의 embed_tokens)을 반환한다.
    # ------------------------------------------------------------------ #
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    # ------------------------------------------------------------------ #
    # lm_head.weight 를 embed_tokens.weight 와 공유(weight tying)한다.
    # ------------------------------------------------------------------ #
    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    # ------------------------------------------------------------------ #
    # hidden → lm_head 로짓(float32) + final logit soft-capping.
    # 몸통 forward 와 분리해, 학습 시 '손실에 필요한 위치만' 골라 호출할 수 있게 한다
    # (전체 시퀀스×vocab FP32 로짓 메모리 폭발 방지 — paligemma2.forward 참고).
    # ------------------------------------------------------------------ #
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden(..., H) → 로짓(..., V), float32 로 승격(수치 안정)
        logits = self.lm_head(hidden_states).float()
        # final logit soft-capping: tanh 로 |logit| 을 cap 이내로 부드럽게 제한(모양 불변)
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits

    # ------------------------------------------------------------------ #
    # 몸통 통과 후 전체 위치의 로짓을 만든다(추론/생성용).
    # 학습(suffix-only) 최적화는 상위 PaliGemma2.forward 에서 compute_logits 로 처리.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(attention_mask, position_ids, inputs_embeds, kv_cache)
        return self.compute_logits(hidden_states)
