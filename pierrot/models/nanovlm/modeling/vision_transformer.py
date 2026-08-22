"""SigLIP2 스타일 ViT 비전 인코더 (스크래치).

패치 임베딩(Conv) → 위치 임베딩 → prenorm 트랜스포머 블록 × N → 최종 LayerNorm.
cls 토큰 없이 모든 패치 토큰을 반환한다(프로젝터가 이를 압축·투영).

모듈/파라미터 이름은 HF SigLIP 키와 매핑되도록 구성되어 있고, from_pretrained 가
공개 SigLIP 가중치를 로드하며 q/k/v 를 하나의 qkv_proj 로 합쳐 넣는다.

텐서 차원 표기:
    B = 이미지(타일) 수, C = 채널(3), H/W = 이미지 크기(예: 512), p = patch_size(16)
    T = num_patches = (H/p)²(예: 1024), D = vit_hidden_dim(768), h = n_heads, hd = head_dim
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTPatchEmbeddings(nn.Module):
    """이미지를 patch_size 격자로 잘라 Conv 로 임베딩하고 위치 임베딩을 더한다."""

    # ------------------------------------------------------------------ #
    # patch_size 커널·스트라이드의 Conv2d 로 비겹침 패치를 뽑고,
    # (cls_flag 면 cls 토큰 +) 학습형 위치 임베딩 파라미터를 준비한다.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg):
        super().__init__()
        self.img_size    = cfg.vit_img_size
        self.patch_size  = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag    = cfg.vit_cls_flag
        self.embd_dim    = cfg.vit_hidden_dim

        self.conv = nn.Conv2d(3, self.embd_dim, kernel_size=self.patch_size,
                              stride=self.patch_size, padding="valid")

        if self.cls_flag:
            self.cls_token          = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, self.embd_dim))
        else:
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches, self.embd_dim))

    # ------------------------------------------------------------------ #
    # (B,3,H,W) → Conv → 패치 평탄화 → (B, num_patches, embd) + 위치 임베딩.
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.conv(x)           # 패치 추출 → (B, D, H/p, W/p)
        x = x.flatten(2)           # 공간축 평탄화 → (B, D, T)
        x = x.transpose(1, 2)      # (B, T, D)  T=num_patches
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)   # (B, 1, D)
            x = torch.cat((cls_token, x), dim=1)                    # (B, T+1, D)
        x = x + self.position_embedding                             # 위치 임베딩 더함 (B, T, D)
        return x


class ViTMultiHeadAttention(nn.Module):
    """양방향 멀티헤드 어텐션(qkv 결합 투영, SDPA 사용)."""

    # ------------------------------------------------------------------ #
    # qkv 를 하나의 선형(3×embd)으로, 출력 투영 별도. head_dim = embd/n_heads.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg):
        super().__init__()
        self.n_heads  = cfg.vit_n_heads
        self.embd_dim = cfg.vit_hidden_dim
        assert self.embd_dim % self.n_heads == 0, "embd_dim 이 헤드 수로 나눠떨어져야 합니다."
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout  = cfg.vit_dropout

        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)
        self.attn_dropout  = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.sdpa = hasattr(F, "scaled_dot_product_attention")

    # ------------------------------------------------------------------ #
    # (B,T,C) → qkv 분리·헤드 분할 → SDPA(양방향, is_causal=False) → 출력 투영.
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()                              # (B, T, C=D)
        qkv     = self.qkv_proj(x)                      # (B, T, 3D)
        q, k, v = qkv.split(C, dim=2)                   # 각 (B, T, D)
        # (B, T, D) → 헤드 분할 → (B, h, T, hd)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.sdpa:
            # 양방향(마스크 없음) → SDPA 가 Flash 커널을 쓸 수 있다. y:(B, h, T, hd)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, h, T, T) 어텐션 점수
            attn = self.attn_dropout(F.softmax(attn, dim=-1))                 # (B, h, T, T)
            y = attn @ v                                                      # (B, h, T, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)   # 헤드 합침 (B, T, D)
        return self.resid_dropout(self.out_proj(y))        # (B, T, D)


class ViTMLP(nn.Module):
    """fc1 → GELU(tanh 근사) → fc2 피드포워드."""

    def __init__(self, cfg):
        super().__init__()
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1           = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim)
        self.fc2           = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim)
        self.dropout       = nn.Dropout(cfg.vit_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,D) → fc1 (B,T,inter) → gelu → fc2 (B,T,D)
        return self.dropout(self.fc2(self.activation_fn(self.fc1(x))))


class ViTBlock(nn.Module):
    """prenorm 잔차 블록: x + attn(ln1(x)); x + mlp(ln2(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.mlp  = ViTMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 전 구간 (B, T, D) 모양 불변. pre-norm 후 잔차로 더함.
        x = x + self.attn(self.ln1(x))     # (B, T, D)
        x = x + self.mlp(self.ln2(x))      # (B, T, D)
        return x


class ViT(nn.Module):
    """SigLIP2 비전 인코더: 패치 임베딩 → 블록 × N → 최종 LayerNorm."""

    # ------------------------------------------------------------------ #
    # 패치 임베딩·드롭아웃·블록 스택·최종 LayerNorm 구성 후 가중치 초기화.
    # ------------------------------------------------------------------ #
    def __init__(self, cfg):
        super().__init__()
        self.cfg             = cfg
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.cls_flag        = cfg.vit_cls_flag
        self.dropout         = nn.Dropout(cfg.vit_dropout)
        self.blocks          = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])
        self.layer_norm      = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    # Linear/Conv normal(0,0.02), LayerNorm(weight=1,bias=0) 초기화.
    # ------------------------------------------------------------------ #
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # ------------------------------------------------------------------ #
    # (B,3,H,W) → 패치 임베딩 → 블록 통과 → 최종 LayerNorm.
    # cls_flag 면 cls 토큰만, 아니면 전체 패치 토큰(B, num_patches, embd) 반환.
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,3,H,W) → 패치임베딩 (B,T,D) → 블록 스택 (B,T,D) → norm (B,T,D)
        x = self.dropout(self.patch_embedding(x))          # (B, T, D)
        for block in self.blocks:
            x = block(x)                                   # (B, T, D)
        if self.cls_flag:
            return self.layer_norm(x[:, 0])                # cls 토큰만 (B, D)
        return self.layer_norm(x)                          # 전체 패치 토큰 (B, T, D)

    # ------------------------------------------------------------------ #
    # 공개 SigLIP 비전 백본 가중치를 로드한다(비전 타워는 스크래치 학습 회피).
    #   - HF SiglipVisionConfig 로 cfg 의 vit_* 를 실제 값으로 덮어씀(in-place)
    #   - HF 키 → 우리 키 매핑으로 복사, q/k/v 는 하나의 qkv_proj 로 concat
    #   - 위치 임베딩은 unsqueeze(0) 로 배치축 추가
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, cfg) -> "ViT":
        from huggingface_hub import hf_hub_download
        import safetensors
        from transformers import SiglipVisionConfig

        hf                 = SiglipVisionConfig.from_pretrained(cfg.vit_model_type)
        cfg.vit_dropout    = hf.attention_dropout
        cfg.vit_hidden_dim = hf.hidden_size
        cfg.vit_img_size   = hf.image_size
        cfg.vit_inter_dim  = hf.intermediate_size
        cfg.vit_ln_eps     = hf.layer_norm_eps
        cfg.vit_n_heads    = hf.num_attention_heads
        cfg.vit_n_blocks   = hf.num_hidden_layers
        cfg.vit_patch_size = hf.patch_size

        model = cls(cfg)
        path  = hf_hub_download(repo_id=cfg.vit_model_type, filename="model.safetensors")
        sd    = model.state_dict()

        mapping = {
            "vision_model.embeddings.patch_embedding.weight": "patch_embedding.conv.weight",
            "vision_model.embeddings.patch_embedding.bias":   "patch_embedding.conv.bias",
            "vision_model.embeddings.position_embedding.weight": "patch_embedding.position_embedding",
            "vision_model.post_layernorm.weight": "layer_norm.weight",
            "vision_model.post_layernorm.bias":   "layer_norm.bias",
        }
        for i in range(cfg.vit_n_blocks):
            p = f"vision_model.encoder.layers.{i}."
            b = f"blocks.{i}."
            mapping.update({
                p + "layer_norm1.weight": b + "ln1.weight",
                p + "layer_norm1.bias":   b + "ln1.bias",
                p + "layer_norm2.weight": b + "ln2.weight",
                p + "layer_norm2.bias":   b + "ln2.bias",
                p + "mlp.fc1.weight": b + "mlp.fc1.weight",
                p + "mlp.fc1.bias":   b + "mlp.fc1.bias",
                p + "mlp.fc2.weight": b + "mlp.fc2.weight",
                p + "mlp.fc2.bias":   b + "mlp.fc2.bias",
                p + "self_attn.out_proj.weight": b + "attn.out_proj.weight",
                p + "self_attn.out_proj.bias":   b + "attn.out_proj.bias",
            })

        with safetensors.safe_open(filename=path, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    t = f.get_tensor(hf_key)
                    if t.shape == sd[our_key].shape:
                        sd[our_key].copy_(t)
                    elif "position_embedding" in hf_key:
                        sd[our_key].copy_(t.unsqueeze(0))
                    else:
                        print(f"[nanovlm:vit] shape mismatch {hf_key}->{our_key}: {t.shape} vs {sd[our_key].shape}")
            # q/k/v 개별 → 결합 qkv_proj 로 수동 concat
            for i in range(cfg.vit_n_blocks):
                p = f"vision_model.encoder.layers.{i}.self_attn."
                qkv_w = torch.cat([f.get_tensor(p + "q_proj.weight"),
                                   f.get_tensor(p + "k_proj.weight"),
                                   f.get_tensor(p + "v_proj.weight")], dim=0)
                sd[f"blocks.{i}.attn.qkv_proj.weight"].copy_(qkv_w)
                qkv_b = torch.cat([f.get_tensor(p + "q_proj.bias"),
                                   f.get_tensor(p + "k_proj.bias"),
                                   f.get_tensor(p + "v_proj.bias")], dim=0)
                sd[f"blocks.{i}.attn.qkv_proj.bias"].copy_(qkv_b)

        model.load_state_dict(sd)
        print(f"[nanovlm:vit] {cfg.vit_model_type} 로드 완료 "
              f"({sum(p.numel() for p in model.parameters()):,} params)")
        return model
