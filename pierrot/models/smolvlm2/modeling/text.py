"""SmolLM2 / Llama 계열 언어 디코더 (스크래치).

구성: RMSNorm + RoPE + Grouped-Query Attention + SwiGLU MLP 의 prenorm 블록.
SmolVLM 백본으로 쓰이므로 입력은 토큰이 아니라 (이미지가 병합된) inputs_embeds 이며,
lm_head 는 디코더 밖(SmolVLM2ForConditionalGeneration)에서 적용된다.

모듈/파라미터 이름은 HF SmolVLM(=Idefics3 의 LlamaModel) 체크포인트 키와 일치한다:
    model.text_model.embed_tokens
    model.text_model.layers.N.self_attn.{q,k,v,o}_proj
    model.text_model.layers.N.mlp.{gate,up,down}_proj
    model.text_model.layers.N.{input_layernorm,post_attention_layernorm}
    model.text_model.norm

텐서 차원 표기:
    B = 배치, T_curr = 현재 시퀀스 길이, T_kv = 키/값 길이(캐시 포함)
    D = hidden_size, hd = head_dim, h = num_attention_heads, kvh = num_key_value_heads
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SmolLM2TextConfig


class SmolLM2RMSNorm(nn.Module):
    """RMS 정규화(평균 차감 없음): x·rsqrt(mean(x²)+eps)·weight."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        irms = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() * irms).to(x.dtype) * self.weight


class SmolLM2RotaryEmbedding(nn.Module):
    """RoPE: position_ids → (cos, sin)."""

    # ------------------------------------------------------------------ #
    # head_dim 절반 주기의 inv_freq 버퍼를 만든다. base = rope_theta.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: SmolLM2TextConfig):
        super().__init__()
        self.dim  = cfg.head_dim
        inv_freq  = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------ #
    # (B, T) position_ids → cos/sin (B, T, hd).
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor):
        b, t  = position_ids.shape
        flat  = position_ids.reshape(-1).float()
        freqs = (flat.unsqueeze(-1) * self.inv_freq.to(position_ids.device).unsqueeze(0)).reshape(b, t, -1)
        emb   = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb), torch.sin(emb)


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
def apply_rotary_pos_embd(q, k, cos, sin, unsqueeze_dim: int = 1):
    # cos/sin 은 float32 로 계산되므로 q 의 dtype 으로 캐스팅한다. 안 그러면 q/k 만 float32 로
    # 승격되어 v(bf16)와 dtype 이 어긋나 SDPA 가 실패한다(HF Llama 와 동일 처리).
    cos = cos.unsqueeze(unsqueeze_dim).to(q.dtype)
    sin = sin.unsqueeze(unsqueeze_dim).to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SmolLM2Attention(nn.Module):
    """Grouped-Query Attention (KV 헤드 공유) + KV 캐시. bias 없음."""

    def __init__(self, cfg: SmolLM2TextConfig):
        super().__init__()
        self.n_heads     = cfg.num_attention_heads
        self.n_kv_heads  = cfg.num_key_value_heads
        self.head_dim    = cfg.head_dim
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.dropout     = cfg.attention_dropout

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

    # ------------------------------------------------------------------ #
    # (B,T,D) → q/k/v 투영·RoPE → (KV 캐시 concat) → GQA 반복 → SDPA → 출력.
    # attention_mask (B, T_kv): 1=참조/0=패딩을 additive(-inf) 마스크로 변환.
    # prefill(T_curr==T_kv>1) 은 causal, decode 는 캐시 전체 참조(causal 불필요).
    # ------------------------------------------------------------------ #
    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        is_prefill = block_kv_cache is None
        B, T_curr, _ = x.size()

        q = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_embd(q, k, cos, sin)

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
            additive = (1.0 - m.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
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


class SmolLM2MLP(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) · up(x)). bias 없음."""

    def __init__(self, cfg: SmolLM2TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SmolLM2DecoderLayer(nn.Module):
    """prenorm 잔차 블록: input_layernorm→attn→+res, post_attention_layernorm→mlp→+res."""

    def __init__(self, cfg: SmolLM2TextConfig):
        super().__init__()
        self.self_attn                = SmolLM2Attention(cfg)
        self.mlp                      = SmolLM2MLP(cfg)
        self.input_layernorm          = SmolLM2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = SmolLM2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        res = x
        x, block_kv_cache = self.self_attn(self.input_layernorm(x), cos, sin, attention_mask, block_kv_cache)
        x = res + x
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, block_kv_cache


class SmolLM2TextModel(nn.Module):
    """SmolLM2 언어 디코더 본체 (inputs_embeds → last_hidden_state)."""

    # ------------------------------------------------------------------ #
    # 임베딩·RoPE·디코더 레이어 스택·최종 RMSNorm 을 구성한다.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: SmolLM2TextConfig):
        super().__init__()
        self.cfg             = cfg
        # padding_idx 는 vocab 범위 안일 때만 전달한다(범위 밖이면 nn.Embedding 이 예외).
        # 최상위 멀티모달 pad_token_id(예: 128002)가 새어 들어와도 안전하도록 방어.
        pad_idx              = cfg.pad_token_id if (cfg.pad_token_id is not None and cfg.pad_token_id < cfg.vocab_size) else None
        self.embed_tokens    = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=pad_idx)
        self.rotary_emb      = SmolLM2RotaryEmbedding(cfg)
        self.layers          = nn.ModuleList([SmolLM2DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm            = SmolLM2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    # ------------------------------------------------------------------ #
    # inputs_embeds(B,T,D) → RoPE(position_ids) → 레이어 스택 → 최종 norm.
    # kv_cache 는 레이어별 dict 리스트(생성 시 재사용). (hidden, kv_cache) 반환.
    # ------------------------------------------------------------------ #
    def forward(self, inputs_embeds, attention_mask=None, position_ids=None, kv_cache=None, start_pos: int = 0):
        B, T_curr, _ = inputs_embeds.size()
        if position_ids is None:
            position_ids = torch.arange(start_pos, start_pos + T_curr, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(position_ids)

        if kv_cache is None:
            kv_cache = [None] * len(self.layers)

        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x, kv_cache[i] = layer(x, cos, sin, attention_mask, kv_cache[i])

        return self.norm(x), kv_cache
