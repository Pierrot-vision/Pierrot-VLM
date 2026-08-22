"""SmolLM2 / Llama 계열 언어 디코더 (스크래치).

구성: RMSNorm + RoPE + Grouped-Query Attention + SwiGLU MLP 의 prenorm 블록.
VLM 백본으로 쓸 때 `lm_use_tokens=False` → 입력이 토큰이 아니라 임베딩이고,
lm_head 는 디코더 밖(VLM)에서 적용된다(이미지 임베딩 병합 경로가 성립).

모듈/파라미터 이름은 HF Llama 키(model.layers.*, model.embed_tokens ...)와 매핑되어
from_pretrained 가 공개 SmolLM2 가중치를 로드하고, 확장된 vocab(추가 66토큰)은
앞부분 복사 + 나머지 normal 초기화로 처리한다.

텐서 차원 표기:
    B  = 배치, T = 현재 시퀀스 길이(T_curr), T_kv = 키/값 길이(캐시 포함)
    D  = lm_hidden_dim, V = lm_vocab_size, hd = head_dim(D/n_heads)
    h  = n_heads(Q 헤드 수), kvh = n_kv_heads(KV 헤드 수), g = h/kvh(GQA 그룹 수)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS 정규화(평균 차감 없음): x·rsqrt(mean(x²)+eps)·weight."""

    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps    = cfg.lm_rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:(B,T,D) → 마지막 축 RMS 로 정규화(모양 불변). irms:(B,T,1)
        irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * irms * self.weight                       # (B, T, D)


class RotaryEmbedding(nn.Module):
    """RoPE: position_ids → (cos, sin). 컨텍스트 초과 시 동적 스케일링."""

    # ------------------------------------------------------------------ #
    # head_dim 절반 주기의 inv_freq 버퍼를 만든다. base = lm_re_base(theta).
    # ------------------------------------------------------------------ #
    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "hidden 이 헤드 수로 나눠떨어져야 합니다."
        self.dim              = cfg.lm_hidden_dim // cfg.lm_n_heads
        self.base             = cfg.lm_re_base
        self.max_seq_len      = cfg.lm_max_position_embeddings
        inv_freq              = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling    = cfg.lm_attn_scaling

    # ------------------------------------------------------------------ #
    # (B, T) position_ids → cos/sin (B, T, dim). 최대 위치가 학습 컨텍스트를
    # 넘으면 inv_freq 를 스케일로 나눠 회전을 더 촘촘히(길이 외삽).
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor):
        # position_ids: (B, T)
        batch_size, seq_len = position_ids.shape
        max_seq  = position_ids.max() + 1
        inv_freq = self.inv_freq / (max_seq / self.original_max_seq_len) \
            if max_seq > self.original_max_seq_len else self.inv_freq       # (hd/2,)

        flat  = position_ids.reshape(-1).float()                            # (B·T,)
        # 위치 × 주파수 외적 → (B, T, hd/2)
        freqs = (flat.unsqueeze(-1) * inv_freq.unsqueeze(0)).reshape(batch_size, seq_len, -1)
        emb   = torch.cat([freqs, freqs], dim=-1)                           # (B, T, hd)
        return torch.cos(emb) * self.attention_scaling, torch.sin(emb) * self.attention_scaling  # 각 (B, T, hd)


# ------------------------------------------------------------------ #
# 마지막 차원을 반으로 나눠 뒤 절반을 부호반전해 앞으로 회전(RoPE 보조).
# ------------------------------------------------------------------ #
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# ------------------------------------------------------------------ #
# q,k 에 RoPE 회전을 적용한다: rotated = q·cos + rotate_half(q)·sin.
# cos/sin 은 헤드축(unsqueeze_dim=1)으로 브로드캐스트.
# ------------------------------------------------------------------ #
def apply_rotary_pos_embd(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LanguageModelGroupedQueryAttention(nn.Module):
    """Grouped-Query Attention: KV 헤드를 여러 Q 헤드가 공유(계산량↓). KV 캐시 지원."""

    # ------------------------------------------------------------------ #
    # q_proj(전체), k/v_proj(KV 헤드분), out_proj. bias 없음. n_kv_groups=Q/KV.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg):
        super().__init__()
        self.n_heads    = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim   = cfg.lm_hidden_dim
        self.dropout    = cfg.lm_dropout
        assert self.n_heads % self.n_kv_heads == 0, "n_heads 가 n_kv_heads 로 나눠떨어져야 합니다."
        assert self.embd_dim % self.n_heads == 0,  "embd_dim 이 헤드 수로 나눠떨어져야 합니다."
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim    = self.embd_dim // self.n_heads

        self.q_proj   = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj   = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj   = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.attn_dropout  = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.sdpa = hasattr(F, "scaled_dot_product_attention")

    # ------------------------------------------------------------------ #
    # (B,T,C) → q/k/v 투영·RoPE → (KV 캐시 concat) → GQA 반복 → SDPA → 출력.
    # attention_mask (B, T_kv): 1=참조/0=패딩을 additive(-inf) 마스크로 변환.
    # prefill(T_curr==T_kv>1) 은 is_causal=True 로 인과성 부여, decode 는 불필요.
    # ------------------------------------------------------------------ #
    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        is_prefill = block_kv_cache is None
        B, T_curr, C = x.size()                          # (B, T_curr, C=D)

        # Q 는 h 헤드, K/V 는 kvh 헤드(적음)로 투영 후 헤드축을 앞으로.
        q_curr = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)     # (B, h,   T_curr, hd)
        k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, kvh, T_curr, hd)
        v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, kvh, T_curr, hd)
        q, k_rot = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)   # q,k 에 RoPE 회전 적용(모양 불변)

        if not is_prefill and block_kv_cache["key"] is not None:
            # 디코드: 캐시된 K/V 에 새 토큰 것을 이어붙임 → T_kv 증가
            k = torch.cat([block_kv_cache["key"], k_rot], dim=2)     # (B, kvh, T_kv, hd)
            v = torch.cat([block_kv_cache["value"], v_curr], dim=2)  # (B, kvh, T_kv, hd)
            block_kv_cache["key"], block_kv_cache["value"] = k, v
        else:
            k, v = k_rot, v_curr                                     # 프리필: 캐시 새로 시작
            block_kv_cache = {"key": k, "value": v}

        # GQA: KV 헤드를 g(=h/kvh)배 복제해 Q 헤드 수(h)에 맞춘다.
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)         # (B, h, T_kv, hd)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)         # (B, h, T_kv, hd)
        T_kv  = k_exp.size(2)

        # 패딩 마스크 (B,1,1,T_kv): pad 키를 -inf. 전부 참조(1)면 0 텐서(무해).
        additive = None
        if attention_mask is not None:
            m        = attention_mask[:, :T_kv]
            additive = (1.0 - m.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
        need_causal = (T_curr == T_kv and T_curr > 1)   # prefill 은 causal 필요, decode(1토큰)는 불필요

        if self.sdpa and x.device.type != "mps" and additive is None:
            # 패딩 없음: SDPA 안전 fast path — attn_mask 없이 is_causal 만 준다.
            y = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=need_causal)
        else:
            # ★ 패딩 있음(좌측패딩): causal 과 padding 을 하나의 additive bias 로 합쳐서 처리한다.
            #   SDPA 에 attn_mask 와 is_causal=True 를 동시에 주면 버전/백엔드별로 오류·모호성이
            #   생기므로 항상 is_causal=False + 합친 bias 로 호출한다.
            bias = additive
            if need_causal:
                upper       = torch.triu(torch.ones(T_curr, T_kv, device=x.device, dtype=torch.bool), diagonal=1)
                causal_bias = torch.zeros(T_curr, T_kv, device=x.device, dtype=q.dtype).masked_fill(
                    upper, torch.finfo(q.dtype).min).view(1, 1, T_curr, T_kv)
                bias = causal_bias if bias is None else bias + causal_bias
                # NaN 방지: 각 쿼리가 최소 자기 자신(대각선)은 보게 한다. 전부 pad 인 쿼리 행이
                # all -inf 가 되어 softmax NaN 이 나는 것을 막는다(그 행 출력은 손실에서 -100 제외).
                eye  = torch.eye(T_curr, T_kv, device=x.device, dtype=torch.bool).view(1, 1, T_curr, T_kv)
                bias = bias.masked_fill(eye, 0.0)

            if self.sdpa and x.device.type != "mps":
                y = F.scaled_dot_product_attention(
                    q, k_exp, v_exp, attn_mask=bias,
                    dropout_p=self.dropout if self.training else 0.0, is_causal=False)
            else:
                attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim)
                if bias is not None:
                    attn = attn + bias
                attn = self.attn_dropout(F.softmax(attn, dim=-1))
                y = attn @ v_exp

        y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
        return self.resid_dropout(self.out_proj(y)), block_kv_cache


class LanguageModelMLP(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) · up(x)). bias 없음."""

    def __init__(self, cfg):
        super().__init__()
        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(cfg.lm_hidden_dim, cfg.lm_inter_dim, bias=False)
        self.up_proj   = nn.Linear(cfg.lm_hidden_dim, cfg.lm_inter_dim, bias=False)
        self.down_proj = nn.Linear(cfg.lm_inter_dim, cfg.lm_hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (B,T,D) → gate/up (B,T,inter) → silu(gate)·up → down (B,T,D)
        return self.down_proj(self.activation_fn(self.gate_proj(x)) * self.up_proj(x))


class LanguageModelBlock(nn.Module):
    """prenorm 잔차 블록: norm1→attn→+res, norm2→mlp→+res."""

    def __init__(self, cfg):
        super().__init__()
        self.mlp   = LanguageModelMLP(cfg)
        self.attn  = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg)
        self.norm2 = RMSNorm(cfg)

    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        # 전 구간 (B, T, D) 모양 불변. pre-norm 후 잔차로 더함.
        res = x
        x, block_kv_cache = self.attn(self.norm1(x), cos, sin, attention_mask, block_kv_cache)  # (B, T, D)
        x = res + x                        # 잔차
        x = x + self.mlp(self.norm2(x))    # (B, T, D)
        return x, block_kv_cache


class LanguageModel(nn.Module):
    """토큰/임베딩 입력 겸용 디코더. VLM 백본 시 lm_use_tokens=False."""

    # ------------------------------------------------------------------ #
    # 임베딩·RoPE·블록 스택·최종 RMSNorm·head 구성. tie_weights 면 head 를
    # token_embedding 과 공유. 이어서 가중치 초기화.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg):
        super().__init__()
        self.cfg           = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd     = RotaryEmbedding(cfg)
        self.blocks          = nn.ModuleList([LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)])
        self.norm            = RMSNorm(cfg)
        self.head            = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    # Linear normal(0,0.02)/Embedding normal(0,0.02)/RMSNorm weight=1 초기화.
    # ------------------------------------------------------------------ #
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    # ------------------------------------------------------------------ #
    # 텍스트 임베딩 테이블을 반환(이미지 임베딩 병합·tie 기준).
    # ------------------------------------------------------------------ #
    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding

    # ------------------------------------------------------------------ #
    # x(토큰이면 임베딩, 아니면 임베딩 그대로) → RoPE(start_pos부터) → 블록 →
    # 최종 norm. lm_use_tokens 면 head 적용해 로짓, 아니면 hidden 반환.
    # kv_cache 는 블록별 dict 리스트(생성 시 재사용). (x, kv_cache) 반환.
    # ------------------------------------------------------------------ #
    def forward(self, x, attention_mask=None, kv_cache=None, start_pos: int = 0):
        # x: 토큰이면 (B, T) → 임베딩 (B, T, D), 임베딩이면 (B, T, D) 그대로
        if self.lm_use_tokens:
            x = self.token_embedding(x)                     # (B, T, D)

        B, T_curr, _ = x.size()                             # (B, T_curr, D)
        # start_pos 부터 T_curr 개의 위치 인덱스(디코드 시 캐시 길이 이후부터)
        position_ids = torch.arange(start_pos, start_pos + T_curr, device=x.device).unsqueeze(0).expand(B, -1)  # (B, T_curr)
        cos, sin     = self.rotary_embd(position_ids)       # 각 (B, T_curr, hd)

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)            # 블록별 KV 캐시 슬롯

        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])

        x = self.norm(x)                                    # 최종 RMSNorm (B, T, D)
        if self.lm_use_tokens:
            x = self.head(x)                                # 로짓 (B, T, V) — VLM 백본 시엔 미적용(hidden 반환)
        return x, kv_cache                                  # (B, T, D 또는 V), 갱신된 kv_cache

    # ------------------------------------------------------------------ #
    # 공개 SmolLM2(HF Llama) 언어 백본 가중치를 로드한다.
    #   - HF AutoConfig 로 cfg 의 lm_* 를 실제 값으로 덮어씀(in-place)
    #   - 우리 vocab(lm_vocab_size)이 원본보다 커야 함(추가 특수토큰). 작으면 오류.
    #   - 샤딩(model.safetensors.index.json)/단일 파일 모두 처리
    #   - 확장 vocab: 앞부분에 원본 임베딩 복사 + 나머지 normal 초기화, head 동기화
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, cfg) -> "LanguageModel":
        import json

        import safetensors
        import torch.nn.init as init
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(cfg.lm_model_type)
        original_vocab_size = hf.vocab_size

        cfg.lm_hidden_dim             = hf.hidden_size
        cfg.lm_inter_dim              = hf.intermediate_size
        cfg.lm_rms_eps                = hf.rms_norm_eps
        cfg.lm_re_base                = hf.rope_theta
        cfg.lm_max_position_embeddings = hf.max_position_embeddings
        if cfg.lm_vocab_size < original_vocab_size:
            raise ValueError(f"cfg.lm_vocab_size({cfg.lm_vocab_size}) < 사전학습 vocab({original_vocab_size})")
        cfg.lm_n_heads    = hf.num_attention_heads
        cfg.lm_n_kv_heads = hf.num_key_value_heads
        cfg.lm_dropout    = hf.attention_dropout
        cfg.lm_n_blocks   = hf.num_hidden_layers

        model = cls(cfg)

        try:
            index_path = hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors.index.json")
            with open(index_path) as f:
                index = json.load(f)
            names = sorted(set(index["weight_map"].values()))
            files = [hf_hub_download(repo_id=cfg.lm_model_type, filename=n) for n in names]
        except EntryNotFoundError:
            files = [hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")]

        sd = model.state_dict()
        mapping = {
            "model.embed_tokens.weight": "token_embedding.weight",
            "model.norm.weight": "norm.weight",
        }
        for i in range(cfg.lm_n_blocks):
            lp = f"model.layers.{i}."
            bp = f"blocks.{i}."
            mapping.update({
                lp + "self_attn.q_proj.weight": bp + "attn.q_proj.weight",
                lp + "self_attn.k_proj.weight": bp + "attn.k_proj.weight",
                lp + "self_attn.v_proj.weight": bp + "attn.v_proj.weight",
                lp + "self_attn.o_proj.weight": bp + "attn.out_proj.weight",
                lp + "mlp.gate_proj.weight": bp + "mlp.gate_proj.weight",
                lp + "mlp.up_proj.weight":   bp + "mlp.up_proj.weight",
                lp + "mlp.down_proj.weight": bp + "mlp.down_proj.weight",
                lp + "input_layernorm.weight":          bp + "norm1.weight",
                lp + "post_attention_layernorm.weight": bp + "norm2.weight",
            })

        has_extended = False
        loaded: set = set()
        for path in files:
            with safetensors.safe_open(filename=path, framework="pt", device="cpu") as f:
                for hf_key, our_key in mapping.items():
                    if our_key in loaded or hf_key not in f.keys() or our_key not in sd:
                        continue
                    t = f.get_tensor(hf_key)
                    if hf_key == "model.embed_tokens.weight" and t.shape[0] != sd[our_key].shape[0]:
                        # 확장 vocab: 원본 임베딩 복사 + 새 토큰 normal 초기화
                        has_extended = True
                        sd[our_key][:t.shape[0]].copy_(t)
                        init.normal_(sd[our_key][t.shape[0]:], mean=0.0, std=0.02)
                        sd["head.weight"].copy_(sd[our_key])
                        print(f"[nanovlm:lm] 임베딩 확장 {t.shape} → {sd[our_key].shape}")
                    elif t.shape == sd[our_key].shape:
                        sd[our_key].copy_(t)
                    else:
                        print(f"[nanovlm:lm] shape mismatch {hf_key}->{our_key}: {t.shape} vs {sd[our_key].shape}")
                    loaded.add(our_key)

        model.load_state_dict(sd)

        # 별도 lm_head 를 가진 백본이면(비-tie) head 도 확장 처리
        if has_extended and "head.weight" in sd:
            for path in files:
                with safetensors.safe_open(filename=path, framework="pt", device="cpu") as f:
                    if "lm_head.weight" in f.keys():
                        lm_head = f.get_tensor("lm_head.weight")
                        if lm_head.shape[0] != sd["head.weight"].shape[0]:
                            sd["head.weight"][:lm_head.shape[0]].copy_(lm_head)
                            init.normal_(sd["head.weight"][lm_head.shape[0]:], mean=0.0, std=0.02)
                            model.load_state_dict(sd)
                        break

        if cfg.lm_tie_weights:
            model.head.weight = model.token_embedding.weight

        print(f"[nanovlm:lm] {cfg.lm_model_type} 로드 완료 "
              f"({sum(p.numel() for p in model.parameters()):,} params)")
        return model
