"""Qwen3.5 하이브리드 언어 디코더 (Gated DeltaNet + gated full attention, 스크래치).

구성: 32개 prenorm 잔차 블록 중 4번째마다 **full attention**, 나머지는
**Gated DeltaNet**(linear attention) 이다(config.layer_types). 두 종류 모두
같은 SwiGLU MLP 를 쓰며, lm_head 는 디코더 밖(Qwen35ForConditionalGeneration)에서
적용된다. 입력은 (이미지가 병합된) inputs_embeds 다.

Qwen3-VL 디코더와 다른 네 가지:
  1) **하이브리드 토큰 믹서** — linear attention 레이어는 어텐션 행렬 없이
     상태(recurrent state) 하나를 갱신하며 시퀀스를 훑는다. 시퀀스가 길어도
     상태 크기가 일정해 KV 캐시가 필요 없다(대신 conv/recurrent 상태를 캐시).
  2) **출력 게이트 어텐션** — q_proj 가 (쿼리‖게이트)를 함께 뽑고, 어텐션 출력에
     sigmoid(gate) 를 곱한 뒤 o_proj 를 지난다(attn_output_gate).
  3) **부분 회전 M-RoPE** — head_dim(256) 중 앞 rotary_dim(64)만 회전하고 나머지는
     통과시킨다. M-RoPE 3축 배분(mrope_section=[11,11,10])은 회전 차원 안에서 한다.
  4) **zero-centered RMSNorm** — 가중치를 0 근처로 저장하고 (1 + weight) 로 곱한다.
     체크포인트 값이 이 규약이므로 일반 RMSNorm 으로 로드하면 조용히 틀린다.

Gated DeltaNet 한 스텝(delta rule + 게이트 감쇠):
    S_t = g_t · S_{t-1} + β_t · k_t ⊗ (v_t − S_{t-1}ᵀ k_t),   o_t = S_tᵀ q_t
  g_t = exp(−exp(A_log)·softplus(a_t + dt_bias)) ∈ (0,1) 는 상태 감쇠(망각),
  β_t = sigmoid(b_t) 는 새 정보의 쓰기 강도다. 입력 q/k/v 는 depthwise 인과 conv
  (커널 4)를 지나고, q/k 는 L2 정규화된다. 출력은 z 게이트로 정규화(RMSNormGated).
  학습/프리필은 청크(64) 단위 병렬식, 디코드는 위 점화식 그대로 계산한다.

모듈/파라미터 이름은 HF Qwen3.5 체크포인트 키와 일치한다:
    model.language_model.embed_tokens
    model.language_model.layers.N.linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a}
    model.language_model.layers.N.linear_attn.{conv1d,dt_bias,A_log,norm,out_proj}
    model.language_model.layers.N.self_attn.{q,k,v,o}_proj / {q,k}_norm
    model.language_model.layers.N.mlp.{gate,up,down}_proj
    model.language_model.layers.N.{input_layernorm,post_attention_layernorm}
    model.language_model.norm

텐서 차원 표기:
    B = 배치, T_curr = 현재 시퀀스 길이, T_kv = 키/값 길이(캐시 포함)
    D = hidden_size, hd = head_dim(256), rd = rotary_dim(64)
    Hk/Hv = linear K/V 헤드 수(16/32), dk/dv = linear 헤드 차원(128/128)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Qwen35TextConfig


class Qwen35RMSNorm(nn.Module):
    """zero-centered RMSNorm: x·rsqrt(mean(x²)+eps)·(1 + weight).

    가중치가 0 을 중심으로 저장된다(초기값 0 = 스케일 1). 체크포인트의 norm 가중치가
    이 규약이라, (1 + weight) 를 빼먹으면 로드는 되지만 출력이 전부 틀린다.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (1.0 + self.weight.float())).type_as(x)


class Qwen35RMSNormGated(nn.Module):
    """DeltaNet 출력 정규화: RMSNorm 후 silu(z) 게이트를 곱한다(norm → gate 순서).

    일반 weight(1 중심) 규약이다 — zero-centered 는 블록 norm 에만 적용된다.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype                                              # x/gate (B·L·Hv, dv)
        xf    = x.float()
        xf    = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)   # RMS 정규화(float32)
        out   = self.weight * xf.to(dtype)                           # 스케일(1 중심 weight)
        out   = out * F.silu(gate.float())                           # ★ norm 후 silu(z) 게이트 (공식 순서)
        return out.to(dtype)


class Qwen35TextRotaryEmbedding(nn.Module):
    """부분 회전 M-RoPE: 3축 position_ids (3, B, T) → (cos, sin) (B, T, rd)."""

    # ------------------------------------------------------------------ #
    # rotary_dim(=head_dim×partial factor) 절반 주기의 inv_freq 와 축별 배분을 준비.
    # cos/sin 은 head_dim 이 아니라 **rotary_dim** 크기다 — 어텐션이 앞 rd 차원만
    # 회전하고 나머지는 통과시킨다.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: Qwen35TextConfig):
        super().__init__()
        self.dim           = cfg.rotary_dim
        self.mrope_section = list(cfg.mrope_section)
        self.interleaved   = cfg.mrope_interleaved
        if sum(self.mrope_section) != self.dim // 2:
            raise ValueError(
                f"mrope_section 합({sum(self.mrope_section)})이 rotary_dim/2({self.dim // 2}) 와 일치해야 합니다."
            )
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------ #
    # 세 축 주파수를 하나의 (…, rd/2) 벡터로 합친다(qwen3vl 과 동일 규칙).
    #   interleaved: [T H W T H W ...] 3칸 간격 교차 배치, 저주파 꼬리는 T 축.
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
    # (3, B, T) position_ids → cos/sin (B, T, rd). 각도 계산은 float32.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        inv_freq = self.inv_freq.to(position_ids.device).float()
        freqs    = position_ids.float().unsqueeze(-1) * inv_freq     # (3, B, T, rd/2)
        combined = self._combine_axes(freqs)                         # (B, T, rd/2)
        emb      = torch.cat([combined, combined], dim=-1)           # (B, T, rd)
        return emb.cos().to(dtype), emb.sin().to(dtype)


# ------------------------------------------------------------------ #
# 마지막 차원을 반으로 나눠 뒤 절반을 부호반전해 앞으로 회전(RoPE 보조).
# ------------------------------------------------------------------ #
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# ------------------------------------------------------------------ #
# 부분 회전 RoPE: 앞 rotary_dim(=cos 마지막 차원)만 회전하고 나머지는 통과.
# cos/sin 은 헤드축(unsqueeze_dim=1)으로 브로드캐스트.
# ------------------------------------------------------------------ #
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)                               # (B, 1, T, rd) — 헤드축 브로드캐스트
    sin = sin.unsqueeze(unsqueeze_dim)
    rd  = cos.shape[-1]                                              # 회전 차원(= head_dim × 0.25 = 64)

    q_rot, q_pass = q[..., :rd], q[..., rd:]                         # (B, h, T, rd) / (B, h, T, hd−rd)
    k_rot, k_pass = k[..., :rd], k[..., rd:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)               # 앞 rd 만 회전
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)   # (B, h, T, hd) 복원


# ------------------------------------------------------------------ #
# FLA 라이브러리와 정렬된 L2 정규화(eps 가 sqrt 안에 들어간다).
# DeltaNet 의 q/k 에 적용 — delta rule 의 안정성(상태 폭주 방지) 핵심.
# ------------------------------------------------------------------ #
def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


# ------------------------------------------------------------------ #
# 청크 병렬 gated delta rule (학습/프리필용, 공식 torch 폴백과 동일 수식).
#
# 시퀀스를 chunk_size(64) 청크로 잘라, 청크 안은 병렬 행렬식으로, 청크 사이는
# 상태(S)를 넘겨 가며 계산한다. β 로 스케일한 k 의 자기상관을 하삼각 역행렬로 풀어
# (I − tril(diag(β)KKᵀ⊙decay))⁻¹ 청크 내 순차 의존성을 한 번에 해소하는 구조다.
# 입력: q/k (B,T,Hv,dk), v (B,T,Hv,dv), g/beta (B,T,Hv)  — q/k 는 이미 Hv 로 복제됨.
# 반환: (출력 (B,T,Hv,dv), 마지막 상태 (B,Hv,dk,dv) 또는 None)
# ------------------------------------------------------------------ #
def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size: int = 64,
                           initial_state=None, output_final_state: bool = False):
    initial_dtype = query.dtype
    query = l2norm(query)                                                    # (B, T, Hv, dk)
    key   = l2norm(key)                                                      # (B, T, Hv, dk)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]                                                                        # q/k (B,Hv,T,dk) · v (B,Hv,T,dv) · β/g (B,Hv,T)

    batch, heads, seq_len, k_dim = key.shape
    v_dim    = value.shape[-1]
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size              # T → C(=64) 배수로 우측 0 패딩
    query    = F.pad(query, (0, 0, 0, pad_size))                             # (B, Hv, T+pad, dk)
    key      = F.pad(key, (0, 0, 0, pad_size))
    value    = F.pad(value, (0, 0, 0, pad_size))
    beta     = F.pad(beta, (0, pad_size))                                    # (B, Hv, T+pad) — pad 는 β=0 → 상태 무영향
    g        = F.pad(g, (0, pad_size))                                       # pad 는 g=0 → 감쇠 exp(0)=1 무영향
    total    = seq_len + pad_size
    query    = query * (k_dim ** -0.5)                                       # 1/√dk 스케일(softmax attn 과 동일 관례)

    v_beta = value * beta.unsqueeze(-1)                                      # (B, Hv, T+pad, dv) = β·v
    k_beta = key * beta.unsqueeze(-1)                                        # (B, Hv, T+pad, dk) = β·k
    # (B, Hv, T+pad, d) → (B, Hv, N, C, d)   N = 청크 수, C = chunk_size
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g    = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)                 # (B, Hv, N, C)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # 청크 내 누적 감쇠와 (I − tril(...))⁻¹ 를 전진대입으로 구성한다.
    g          = g.cumsum(dim=-1)                                            # (B, Hv, N, C) 청크 내 log-감쇠 누적합
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()   # (B,Hv,N,C,C) exp(g_i−g_j), i≥j
    attn       = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)  # (B,Hv,N,C,C) −tril(β·KKᵀ⊙decay)
    for i in range(1, chunk_size):                                           # 전진대입으로 (I−tril)⁻¹−I 를 행 단위 누적
        row = attn[..., i, :i].clone()                                       # (B, Hv, N, i)
        sub = attn[..., :i, :i].clone()                                      # (B, Hv, N, i, i)
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)  # (B,Hv,N,C,C) = (I−tril)⁻¹ 완성

    value      = attn @ v_beta                                               # (B,Hv,N,C,dv) 청크 내 의존 해소된 β·v
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))                     # (B,Hv,N,C,dk) 상태 읽기용 계수
    state = (
        torch.zeros(batch, heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )                                                                        # (B, Hv, dk, dv)
    out  = torch.zeros_like(value)                                           # (B, Hv, N, C, dv)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # 청크 순회: 청크 내 어텐션 + 이전 상태 기여 → 상태 갱신.
    for i in range(total // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]         # (B,Hv,C,dk) ·2 / (B,Hv,C,dv)
        attn_i     = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)  # (B,Hv,C,C) 청크 내 인과
        v_prime    = k_cumdecay[:, :, i] @ state                             # (B,Hv,C,dv) 이전 상태가 이미 기억한 값
        v_new      = v_i - v_prime                                           # (B,Hv,C,dv) delta rule: 차이만 기록
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state               # (B,Hv,C,dv) 청크 밖(상태) 기여
        out[:, :, i] = attn_inter + attn_i @ v_new                           # (B,Hv,C,dv) 상태 + 청크 내 합
        state = state * g[:, :, i, -1, None, None].exp() + (                 # (B,Hv,dk,dv) 청크 끝까지 감쇠 후
            k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]       #   각 토큰 잔여 감쇠 반영해 kᵀ·v_new 적립
        ).transpose(-1, -2) @ v_new

    if not output_final_state:
        state = None
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])[:, :, :seq_len]  # (B, Hv, T, dv) pad 제거
    return out.transpose(1, 2).contiguous().to(initial_dtype), state         # (B, T, Hv, dv), (B, Hv, dk, dv)


# ------------------------------------------------------------------ #
# 순차 gated delta rule (디코드용): 점화식을 토큰 단위로 그대로 계산한다.
#     S_t = g_t·S_{t-1} + β_t·k_t ⊗ (v_t − S_{t-1}ᵀk_t),  o_t = S_tᵀq_t
# ------------------------------------------------------------------ #
def recurrent_gated_delta_rule(query, key, value, g, beta,
                               initial_state=None, output_final_state: bool = False):
    initial_dtype = query.dtype
    query = l2norm(query)                                                    # (B, T, Hv, dk)
    key   = l2norm(key)                                                      # (B, T, Hv, dk)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]                                                                        # q/k (B,Hv,T,dk) · v (B,Hv,T,dv) · β/g (B,Hv,T)

    batch, heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    query = query * (k_dim ** -0.5)                                          # 1/√dk 스케일

    out   = torch.zeros(batch, heads, seq_len, v_dim, dtype=value.dtype, device=value.device)
    state = (
        torch.zeros(batch, heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )                                                                        # (B, Hv, dk, dv)

    for i in range(seq_len):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]         # (B,Hv,dk) ·2 / (B,Hv,dv)
        g_t    = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)                # (B, Hv, 1, 1) 감쇠 ∈ (0,1)
        beta_t = beta[:, :, i].unsqueeze(-1)                                 # (B, Hv, 1) 쓰기 강도

        state  = state * g_t                                                 # 망각: S ← g·S
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)                     # (B, Hv, dv) = Sᵀk (이미 기억한 값)
        delta  = (v_t - kv_mem) * beta_t                                     # (B, Hv, dv) 새 정보와의 차이만
        state  = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)             # 쓰기: S ← S + k ⊗ delta
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)               # (B, Hv, dv) = Sᵀq (읽기)

    if not output_final_state:
        state = None
    return out.transpose(1, 2).contiguous().to(initial_dtype), state         # (B, T, Hv, dv), (B, Hv, dk, dv)


class Qwen35GatedDeltaNet(nn.Module):
    """Gated DeltaNet 토큰 믹서 (linear attention 레이어).

    캐시는 KV 가 아니라 두 상태다(길이와 무관한 고정 크기):
      - conv      : depthwise 인과 conv 의 직전 K-1 개 입력 (B, conv_dim, K-1)
      - recurrent : delta rule 상태 (B, Hv, dk, dv)
    """

    def __init__(self, cfg: Qwen35TextConfig):
        super().__init__()
        self.num_v_heads = cfg.linear_num_value_heads
        self.num_k_heads = cfg.linear_num_key_heads
        self.head_k_dim  = cfg.linear_key_head_dim
        self.head_v_dim  = cfg.linear_value_head_dim
        self.key_dim     = cfg.linear_key_dim
        self.value_dim   = cfg.linear_value_dim
        self.conv_dim    = cfg.linear_conv_dim
        self.kernel_size = cfg.linear_conv_kernel_dim

        # q‖k‖v 를 한 번에 뽑아 depthwise 인과 conv 로 지역 문맥을 섞는다(bias 없음).
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z   = nn.Linear(cfg.hidden_size, self.value_dim, bias=False)   # 출력 게이트
        self.in_proj_b   = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False) # β(쓰기 강도)
        self.in_proj_a   = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False) # 감쇠 입력
        self.conv1d      = nn.Conv1d(
            self.conv_dim, self.conv_dim, kernel_size=self.kernel_size,
            groups=self.conv_dim, padding=self.kernel_size - 1, bias=False,
        )

        # 감쇠 파라미터: g = exp(−exp(A_log)·softplus(a + dt_bias)) ∈ (0,1).
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log   = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log())

        self.norm     = Qwen35RMSNormGated(self.head_v_dim, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, cfg.hidden_size, bias=False)

    # ------------------------------------------------------------------ #
    # 프리필 conv 캐시: 원시 입력 mixed (B, conv_dim, L) 에서 '샘플별 마지막 K-1 개
    # 실토큰' 열을 모은다. 우측 패딩이면 끝의 pad 가 실제 토큰 이력을 밀어내므로
    # 단순 mixed[:, :, -(K-1):] 는 오염된다 — 유효 길이 기준으로 gather 한다.
    # 유효 길이 < K-1 이면 부족분을 0 으로 채운다(무패딩 F.pad 경로와 동일 결과).
    # ------------------------------------------------------------------ #
    def _conv_state(self, mixed: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, _, L = mixed.shape
        Km1     = self.kernel_size - 1
        lengths = (pad_mask.sum(-1).long() if pad_mask is not None
                   else torch.full((B,), L, dtype=torch.long, device=mixed.device))     # (B,) 실토큰 수
        pos   = lengths.view(B, 1) - Km1 + torch.arange(Km1, device=mixed.device).view(1, Km1)  # (B, K-1)
        valid = pos >= 0                                                     # 음수 = 이력 부족 → 0 채움
        state = mixed.gather(2, pos.clamp(min=0).view(B, 1, Km1).expand(-1, self.conv_dim, -1))  # (B, conv_dim, K-1)
        return state * valid.view(B, 1, Km1).to(state.dtype)

    # ------------------------------------------------------------------ #
    # (B,T,D) → q/k/v 인과 conv → delta rule(청크 or 순차) → z 게이트 norm → 출력.
    # attention_mask (B,T): 우측 패딩을 0 으로 지워 conv/상태 오염을 막고,
    # pad 스텝의 감쇠(g)도 차단해 캐시 상태를 단독 실행과 일치시킨다.
    # cache dict 가 오면 conv/recurrent 상태를 읽고 갱신한다(디코드).
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                cache: Optional[Dict[str, torch.Tensor]] = None):
        # 패딩 위치를 0 으로 — conv 와 recurrence 가 pad 값을 상태에 싣지 않게 한다.
        pad_mask = None                                                      # (B, L) 1=실토큰, 0=pad
        if attention_mask is not None and attention_mask.shape[-1] == x.shape[1] and x.shape[1] > 1:
            pad_mask = attention_mask.to(x.dtype)
            x = x * pad_mask[:, :, None]

        B, L, _ = x.shape
        decode  = cache is not None and cache.get("recurrent") is not None and L == 1

        mixed = self.in_proj_qkv(x).transpose(1, 2)                          # (B, conv_dim, L)  q‖k‖v 융합
        z     = self.in_proj_z(x).reshape(B, L, self.num_v_heads, self.head_v_dim)  # (B, L, Hv, dv) 출력 게이트
        b     = self.in_proj_b(x)                                            # (B, L, Hv) → β 원료
        a     = self.in_proj_a(x)                                            # (B, L, Hv) → 감쇠 원료

        if decode:
            # 직전 K-1 입력 + 현재 1개 → 폭 K → 인과 conv 출력 1개.
            conv_in       = torch.cat([cache["conv"], mixed], dim=-1)        # (B, conv_dim, K)
            cache["conv"] = conv_in[:, :, -(self.kernel_size - 1):]          # 다음 스텝용 K-1 개 갱신
            mixed = F.conv1d(conv_in, self.conv1d.weight, self.conv1d.bias,
                             padding=0, groups=self.conv_dim)                # (B, conv_dim, 1)
        else:
            if cache is not None:                                            # 프리필: 마지막 K-1 개 '실토큰' 입력을 상태로
                cache["conv"] = self._conv_state(mixed, pad_mask)            # (B, conv_dim, K-1) 샘플별 gather
            mixed = self.conv1d(mixed)[:, :, :L]                             # 좌측 K-1 패딩 conv → 앞 L 개(인과)
        mixed = F.silu(mixed).transpose(1, 2)                                # (B, L, conv_dim)
        if pad_mask is not None:
            # ★ conv 후 재마스킹: 인과 conv 는 pad 위치에서도 직전 실토큰 window 를 보므로
            #   pad 의 q/k/v 가 0 이 아니게 된다 → 그대로 두면 pad 스텝이 상태에 '쓰기'를
            #   한다. 여기서 0 으로 지워야 (감쇠 차단과 함께) 상태가 단독 실행과 일치한다.
            mixed = mixed * pad_mask[:, :, None]                             # (B, L, conv_dim) pad 행 0

        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(B, L, self.num_k_heads, self.head_k_dim)       # (B, L, Hk, dk)
        key   = key.reshape(B, L, self.num_k_heads, self.head_k_dim)         # (B, L, Hk, dk)
        value = value.reshape(B, L, self.num_v_heads, self.head_v_dim)       # (B, L, Hv, dv)

        beta = b.sigmoid()                                                   # (B, L, Hv) 쓰기 강도 ∈ (0,1)
        # fp16 에서 A 가 -inf 가 되는 것을 막기 위해 float 로 계산한다.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias) # (B, L, Hv) log-감쇠 ≤ 0
        if pad_mask is not None:
            # ★ pad 스텝의 감쇠 차단(우측 패딩 배치 생성의 핵심): 입력을 0 으로 지워도
            #   g 는 dt_bias 때문에 0 이 아니라서 pad 스텝마다 상태가 감쇠한다.
            #   g=0(감쇠 exp(0)=1) 로 만들면 쓰기(k=v=0)와 함께 pad 스텝이 상태에 완전
            #   무영향이 되어, recurrent 상태가 단독(B=1) 실행과 정확히 일치한다.
            g = g * pad_mask[:, :, None].to(g.dtype)                         # (B, L, Hv) pad 위치 g=0
        if self.num_v_heads // self.num_k_heads > 1:                         # V 헤드 수에 맞춰 복제
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)   # (B, L, Hv, dk)
            key   = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)     # (B, L, Hv, dk)

        initial_state = cache.get("recurrent") if cache is not None else None              # (B, Hv, dk, dv)|None
        rule          = recurrent_gated_delta_rule if decode else chunk_gated_delta_rule
        out, state    = rule(query, key, value, g=g, beta=beta,
                             initial_state=initial_state, output_final_state=cache is not None)
        if cache is not None:                                                # out (B, L, Hv, dv)
            cache["recurrent"] = state                                       # (B, Hv, dk, dv) 고정 크기

        out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))  # (B·L·Hv, dv) norm→silu(z) 게이트
        out = out.reshape(B, L, self.value_dim)                              # (B, L, Hv·dv=4096)
        return self.out_proj(out)                                            # (B, L, D)


class Qwen35TextAttention(nn.Module):
    """출력 게이트 + QK-Norm + 부분 회전 M-RoPE 를 쓰는 GQA (full attention 레이어)."""

    def __init__(self, cfg: Qwen35TextConfig):
        super().__init__()
        self.n_heads     = cfg.num_attention_heads
        self.n_kv_heads  = cfg.num_key_value_heads
        self.head_dim    = cfg.head_dim
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.dropout     = cfg.attention_dropout

        bias = cfg.attention_bias
        # ★ q_proj 출력이 2배 — 헤드마다 (쿼리 hd ‖ 게이트 hd) 를 함께 뽑는다.
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim * 2, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=bias)
        self.q_norm = Qwen35RMSNorm(self.head_dim, cfg.rms_norm_eps)   # 헤드 차원에서만
        self.k_norm = Qwen35RMSNorm(self.head_dim, cfg.rms_norm_eps)

    # ------------------------------------------------------------------ #
    # (B,T,D) → (쿼리‖게이트) 분리·QK-Norm·부분 RoPE → (KV 캐시 concat) → SDPA
    # → sigmoid(gate) 곱 → o_proj. 마스크 처리 규칙은 qwen3vl 과 동일하다.
    # ------------------------------------------------------------------ #
    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        B, T_curr, _ = x.size()

        qg      = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim * 2)  # (B, T, h, 2hd) 쿼리‖게이트
        q, gate = qg.chunk(2, dim=-1)                                # 헤드별 (쿼리 hd, 게이트 hd)
        gate    = gate.reshape(B, T_curr, self.n_heads * self.head_dim)             # (B, T, h·hd)

        q = self.q_norm(q).transpose(1, 2)                                          # (B, h, T, hd)
        k = self.k_norm(self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim)).transpose(1, 2)  # (B, kvh, T, hd)
        v = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)               # (B, kvh, T, hd)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                 # 앞 rd(=64)차원만 회전, 뒤는 통과

        # 프리필은 캐시 dict 가 있어도 key 가 None 이다(new_cache 규약) — 디코드만 concat.
        if block_kv_cache is not None and block_kv_cache.get("key") is not None:
            k = torch.cat([block_kv_cache["key"], k], dim=2)                        # (B, kvh, T_kv, hd)
            v = torch.cat([block_kv_cache["value"], v], dim=2)
        block_kv_cache = {"key": k, "value": v}

        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)                        # (B, h, T_kv, hd) GQA 복제
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)
        T_kv  = k_exp.size(2)

        additive = None
        if attention_mask is not None:
            m        = attention_mask[:, :T_kv]
            # SDPA 는 attn_mask dtype 이 쿼리와 같아야 한다 — bf16 학습에서 float() 쓰면 죽는다.
            additive = (1.0 - m.unsqueeze(1).unsqueeze(2).to(q.dtype)) * torch.finfo(q.dtype).min
        need_causal = (T_curr == T_kv and T_curr > 1)

        if additive is None:
            y = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=need_causal)
        else:
            bias = additive
            if need_causal:
                upper       = torch.triu(torch.ones(T_curr, T_kv, device=x.device, dtype=torch.bool), diagonal=1)
                causal_bias = torch.zeros(T_curr, T_kv, device=x.device, dtype=q.dtype).masked_fill(
                    upper, torch.finfo(q.dtype).min).view(1, 1, T_curr, T_kv)
                bias = bias + causal_bias
                eye  = torch.eye(T_curr, T_kv, device=x.device, dtype=torch.bool).view(1, 1, T_curr, T_kv)
                bias = bias.masked_fill(eye, 0.0)
            y = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=bias,
                dropout_p=self.dropout if self.training else 0.0, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T_curr, -1)       # (B, T, h·hd)
        y = y * torch.sigmoid(gate)                                  # ★ 출력 게이트 (B, T, h·hd)
        return self.o_proj(y), block_kv_cache                        # (B, T, D)


class Qwen35TextMLP(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) · up(x)). bias 없음."""

    def __init__(self, cfg: Qwen35TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35TextDecoderLayer(nn.Module):
    """prenorm 잔차 블록. layer_types[i] 에 따라 토큰 믹서가 갈린다:
    linear_attention → linear_attn(Gated DeltaNet), full_attention → self_attn."""

    def __init__(self, cfg: Qwen35TextConfig, layer_idx: int):
        super().__init__()
        self.block_type = cfg.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(cfg)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen35TextAttention(cfg)
        else:
            raise ValueError(f"알 수 없는 layer_type: {self.block_type}")
        self.mlp                      = Qwen35TextMLP(cfg)
        self.input_layernorm          = Qwen35RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    # ------------------------------------------------------------------ #
    # cache: full attention 은 {"key","value"}, linear attention 은
    # {"conv","recurrent"} — 레이어 종류에 따라 다른 상태를 주고받는다.
    # ------------------------------------------------------------------ #
    def forward(self, x, cos, sin, attention_mask=None, cache=None):
        res = x
        h   = self.input_layernorm(x)
        if self.block_type == "linear_attention":
            h = self.linear_attn(h, attention_mask=attention_mask, cache=cache)
        else:
            h, cache = self.self_attn(h, cos, sin, attention_mask, cache)
        x = res + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache


class Qwen35TextModel(nn.Module):
    """Qwen3.5 하이브리드 디코더 본체 (inputs_embeds → last_hidden_state)."""

    # ------------------------------------------------------------------ #
    # 임베딩·부분 회전 M-RoPE·하이브리드 레이어 스택·최종 zero-centered RMSNorm.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: Qwen35TextConfig):
        super().__init__()
        self.cfg          = cfg
        pad_idx           = cfg.pad_token_id if (cfg.pad_token_id is not None and cfg.pad_token_id < cfg.vocab_size) else None
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=pad_idx)
        self.rotary_emb   = Qwen35TextRotaryEmbedding(cfg)
        self.layers       = nn.ModuleList([
            Qwen35TextDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)
        ])
        self.norm = Qwen35RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    # ------------------------------------------------------------------ #
    # 디코드용 빈 캐시를 만든다. full attention 레이어는 KV, linear attention
    # 레이어는 conv/recurrent 상태 자리를 갖는다(레이어 종류별로 다른 dict).
    # ------------------------------------------------------------------ #
    def new_cache(self) -> List[Dict]:
        return [
            {"key": None, "value": None} if t == "full_attention" else {"conv": None, "recurrent": None}
            for t in self.cfg.layer_types
        ]

    # ------------------------------------------------------------------ #
    # inputs_embeds(B,T,D) → M-RoPE(position_ids) → 하이브리드 스택 → 최종 norm.
    # position_ids 는 (3,B,T) 또는 (B,T). kv_cache 는 new_cache() 형식의 리스트.
    # (hidden, kv_cache) 반환 — qwen3vl 디코더와 같은 계약이다.
    # ------------------------------------------------------------------ #
    def forward(self, inputs_embeds, attention_mask=None, position_ids=None, kv_cache=None,
                start_pos: int = 0):
        B, T_curr, _ = inputs_embeds.size()
        if position_ids is None:
            position_ids = torch.arange(
                start_pos, start_pos + T_curr, device=inputs_embeds.device
            ).unsqueeze(0).expand(B, -1)                                     # (B, T) 텍스트 전용 위치
        cos, sin = self.rotary_emb(position_ids, inputs_embeds.dtype)        # (B, T, rd) ×2 — full 층만 사용

        if kv_cache is None:
            kv_cache = [None] * len(self.layers)

        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x, kv_cache[i] = layer(x, cos, sin, attention_mask, kv_cache[i])

        return self.norm(x), kv_cache
