"""Qwen3 언어 디코더 (M-RoPE + QK-Norm + GQA, 스크래치).

구성: RMSNorm + 3D RoPE(M-RoPE) + Grouped-Query Attention + SwiGLU MLP 의 prenorm 블록.
Qwen3-VL 백본으로 쓰이므로 입력은 토큰이 아니라 (이미지가 병합된) inputs_embeds 이며,
lm_head 는 디코더 밖(Qwen3VLForConditionalGeneration)에서 적용된다.

다른 Llama 계열 디코더와 다른 두 가지:
  1) **QK-Norm** — q/k 를 헤드 차원(head_dim)에서 RMSNorm 한 뒤 RoPE 를 적용한다
     (Qwen3 고유. 학습 안정화용이며 파라미터가 실제로 체크포인트에 있다).
  2) **M-RoPE** — 위치가 스칼라가 아니라 (시간 t, 높이 h, 너비 w) 3축이다.
     head_dim/2 개 주파수를 mrope_section=[24,20,20] 으로 세 축에 배분하며,
     Qwen3-VL 은 청크가 아닌 **interleaved**([THWTHW...]) 배치를 쓴다.

또 하나의 특징이 **DeepStack** 이다. 비전 중간층 특징을 앞쪽 디코더 레이어 출력의
이미지 토큰 위치에 더한다(= 시각 정보를 여러 깊이에서 재주입). 이 주입은 모델 본체가
아니라 forward 인자(visual_pos_masks / deepstack_visual_embeds)로 들어온다.

모듈/파라미터 이름은 HF Qwen3-VL 체크포인트 키와 일치한다:
    model.language_model.embed_tokens
    model.language_model.layers.N.self_attn.{q,k,v,o}_proj / {q,k}_norm
    model.language_model.layers.N.mlp.{gate,up,down}_proj
    model.language_model.layers.N.{input_layernorm,post_attention_layernorm}
    model.language_model.norm

텐서 차원 표기:
    B = 배치, T_curr = 현재 시퀀스 길이, T_kv = 키/값 길이(캐시 포함)
    D = hidden_size, hd = head_dim, h = num_attention_heads, kvh = num_key_value_heads
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Qwen3VLTextConfig


class Qwen3VLRMSNorm(nn.Module):
    """RMS 정규화(평균 차감 없음): x·rsqrt(mean(x²)+eps)·weight."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        irms = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() * irms).to(x.dtype) * self.weight


class Qwen3VLTextRotaryEmbedding(nn.Module):
    """M-RoPE: 3축 position_ids (3, B, T) → (cos, sin) (B, T, hd)."""

    # ------------------------------------------------------------------ #
    # head_dim 절반 주기의 inv_freq 버퍼와 축별 주파수 배분(mrope_section)을 준비한다.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: Qwen3VLTextConfig):
        super().__init__()
        self.dim           = cfg.head_dim
        self.mrope_section = list(cfg.mrope_section)
        self.interleaved   = cfg.mrope_interleaved
        if sum(self.mrope_section) != self.dim // 2:
            raise ValueError(
                f"mrope_section 합({sum(self.mrope_section)})이 head_dim/2({self.dim // 2}) 와 일치해야 합니다."
            )
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------ #
    # 세 축 주파수를 하나의 (…, hd/2) 벡터로 합친다.
    #   interleaved(Qwen3-VL 기본): [T H W T H W ...] 로 3칸 간격 교차 배치.
    #     축 d(H=1,W=2)는 앞쪽 mrope_section[d]×3 구간에서 offset d 위치를 가져가고,
    #     나머지(뒤쪽 저주파 포함)는 전부 T 축이 차지한다 → 주파수 연속성이 유지된다.
    #   chunked(Qwen2-VL 방식): [TT..T HH..H WW..W] 로 구간을 통째로 나눈다.
    # ------------------------------------------------------------------ #
    def _combine_axes(self, freqs: torch.Tensor) -> torch.Tensor:
        if not self.interleaved:
            out, start = [], 0
            for axis, width in enumerate(self.mrope_section):
                out.append(freqs[axis, ..., start:start + width])
                start += width
            return torch.cat(out, dim=-1)

        combined = freqs[0].clone()                                  # 기본은 전부 T 축
        for axis, offset in enumerate((1, 2), start=1):              # H, W 축만 덮어쓴다
            length = self.mrope_section[axis] * 3
            idx    = slice(offset, length, 3)
            combined[..., idx] = freqs[axis, ..., idx]
        return combined

    # ------------------------------------------------------------------ #
    # (3, B, T) position_ids → cos/sin (B, T, hd). 2D 로 들어오면 세 축 모두 같은
    # 위치로 확장한다(텍스트 전용 입력 대비). 각도 계산은 float32 로 한다.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        inv_freq = self.inv_freq.to(position_ids.device).float()
        freqs    = position_ids.float().unsqueeze(-1) * inv_freq     # (3, B, T, hd/2)
        combined = self._combine_axes(freqs)                         # (B, T, hd/2)
        emb      = torch.cat([combined, combined], dim=-1)           # (B, T, hd)
        return emb.cos().to(dtype), emb.sin().to(dtype)


# ------------------------------------------------------------------ #
# 마지막 차원을 반으로 나눠 뒤 절반을 부호반전해 앞으로 회전(RoPE 보조).
# ------------------------------------------------------------------ #
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# ------------------------------------------------------------------ #
# q,k 에 RoPE 회전을 적용한다: rotated = x·cos + rotate_half(x)·sin.
# cos/sin 은 헤드축(unsqueeze_dim=1)으로 브로드캐스트.
# ------------------------------------------------------------------ #
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Qwen3VLTextAttention(nn.Module):
    """QK-Norm 을 적용한 Grouped-Query Attention + KV 캐시."""

    def __init__(self, cfg: Qwen3VLTextConfig):
        super().__init__()
        self.n_heads     = cfg.num_attention_heads
        self.n_kv_heads  = cfg.num_key_value_heads
        self.head_dim    = cfg.head_dim
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.dropout     = cfg.attention_dropout

        bias = cfg.attention_bias
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=bias)
        # ★ Qwen3 고유: 헤드 차원에서만 RMSNorm (레이어 전체가 아니라 head_dim 크기).
        self.q_norm = Qwen3VLRMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = Qwen3VLRMSNorm(self.head_dim, cfg.rms_norm_eps)

    # ------------------------------------------------------------------ #
    # (B,T,D) → q/k/v 투영·QK-Norm·RoPE → (KV 캐시 concat) → GQA 반복 → SDPA → 출력.
    # attention_mask (B, T_kv): 1=참조/0=패딩을 additive(-inf) 마스크로 변환.
    # prefill(T_curr==T_kv>1) 은 causal, decode 는 캐시 전체 참조(causal 불필요).
    # ------------------------------------------------------------------ #
    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        is_prefill   = block_kv_cache is None
        B, T_curr, _ = x.size()

        q = self.q_norm(self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if not is_prefill and block_kv_cache["key"] is not None:
            k = torch.cat([block_kv_cache["key"], k], dim=2)
            v = torch.cat([block_kv_cache["value"], v], dim=2)
        block_kv_cache = {"key": k, "value": v}

        # GQA: KV 헤드를 g(=h/kvh)배 복제해 Q 헤드 수에 맞춘다.
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)
        T_kv  = k_exp.size(2)

        additive = None
        if attention_mask is not None:
            m        = attention_mask[:, :T_kv]
            # SDPA 는 attn_mask dtype 이 쿼리와 같아야 한다 — bf16 학습에서 float() 쓰면 죽는다.
            additive = (1.0 - m.unsqueeze(1).unsqueeze(2).to(q.dtype)) * torch.finfo(q.dtype).min
        need_causal = (T_curr == T_kv and T_curr > 1)

        if additive is None:
            # 패딩 없음: SDPA 안전 fast path (attn_mask 없이 is_causal 만).
            y = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=need_causal)
        else:
            # 패딩 있음: causal 과 padding 을 하나의 additive bias 로 합쳐서 처리.
            bias = additive
            if need_causal:
                upper       = torch.triu(torch.ones(T_curr, T_kv, device=x.device, dtype=torch.bool), diagonal=1)
                causal_bias = torch.zeros(T_curr, T_kv, device=x.device, dtype=q.dtype).masked_fill(
                    upper, torch.finfo(q.dtype).min).view(1, 1, T_curr, T_kv)
                bias = bias + causal_bias
                # NaN 방지: 각 쿼리가 최소 자기 자신(대각선)은 보게 한다(전부 pad 인 행 방지).
                eye  = torch.eye(T_curr, T_kv, device=x.device, dtype=torch.bool).view(1, 1, T_curr, T_kv)
                bias = bias.masked_fill(eye, 0.0)
            y = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=bias,
                dropout_p=self.dropout if self.training else 0.0, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T_curr, -1)
        return self.o_proj(y), block_kv_cache


class Qwen3VLTextMLP(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) · up(x)). bias 없음."""

    def __init__(self, cfg: Qwen3VLTextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLTextDecoderLayer(nn.Module):
    """prenorm 잔차 블록: input_layernorm→attn→+res, post_attention_layernorm→mlp→+res."""

    def __init__(self, cfg: Qwen3VLTextConfig):
        super().__init__()
        self.self_attn                = Qwen3VLTextAttention(cfg)
        self.mlp                      = Qwen3VLTextMLP(cfg)
        self.input_layernorm          = Qwen3VLRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        res = x
        x, block_kv_cache = self.self_attn(self.input_layernorm(x), cos, sin, attention_mask, block_kv_cache)
        x = res + x
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, block_kv_cache


class Qwen3VLTextModel(nn.Module):
    """Qwen3 언어 디코더 본체 (inputs_embeds → last_hidden_state) + DeepStack 주입."""

    # ------------------------------------------------------------------ #
    # 임베딩·M-RoPE·디코더 레이어 스택·최종 RMSNorm 을 구성한다.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: Qwen3VLTextConfig):
        super().__init__()
        self.cfg          = cfg
        # padding_idx 는 vocab 범위 안일 때만 전달한다(범위 밖이면 nn.Embedding 이 예외).
        pad_idx           = cfg.pad_token_id if (cfg.pad_token_id is not None and cfg.pad_token_id < cfg.vocab_size) else None
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=pad_idx)
        self.rotary_emb   = Qwen3VLTextRotaryEmbedding(cfg)
        self.layers       = nn.ModuleList([Qwen3VLTextDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm         = Qwen3VLRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    # ------------------------------------------------------------------ #
    # DeepStack 주입: 이미지 토큰 위치(visual_pos_masks)의 hidden 에 비전 중간층
    # 특징을 더한다. visual_embeds 행 순서는 마스크의 (배치, 시퀀스) 순회 순서와
    # 같아야 하며, 프로세서/병합이 그 순서를 보장한다.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _deepstack_add(hidden, visual_pos_masks, visual_embeds):
        hidden = hidden.clone()
        hidden[visual_pos_masks, :] = hidden[visual_pos_masks, :] + visual_embeds.to(hidden.dtype)
        return hidden

    # ------------------------------------------------------------------ #
    # inputs_embeds(B,T,D) → M-RoPE(position_ids) → 레이어 스택 → 최종 norm.
    # position_ids 는 (3,B,T)(시간/높이/너비) 또는 (B,T)(텍스트 전용) 이다.
    # deepstack_visual_embeds 가 있으면 앞쪽 N개 레이어 출력에 순서대로 주입한다.
    # kv_cache 는 레이어별 dict 리스트(생성 시 재사용). (hidden, kv_cache) 반환.
    # ------------------------------------------------------------------ #
    def forward(self, inputs_embeds, attention_mask=None, position_ids=None, kv_cache=None,
                start_pos: int = 0, visual_pos_masks=None,
                deepstack_visual_embeds: Optional[List[torch.Tensor]] = None):
        B, T_curr, _ = inputs_embeds.size()
        if position_ids is None:
            position_ids = torch.arange(
                start_pos, start_pos + T_curr, device=inputs_embeds.device
            ).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(position_ids, inputs_embeds.dtype)

        if kv_cache is None:
            kv_cache = [None] * len(self.layers)

        n_deepstack = len(deepstack_visual_embeds) if deepstack_visual_embeds is not None else 0

        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x, kv_cache[i] = layer(x, cos, sin, attention_mask, kv_cache[i])

            if i < n_deepstack and visual_pos_masks is not None:
                x = self._deepstack_add(x, visual_pos_masks, deepstack_visual_embeds[i])

        return self.norm(x), kv_cache
