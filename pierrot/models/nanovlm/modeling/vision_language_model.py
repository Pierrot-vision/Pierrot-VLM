"""nanoVLM 본체: 비전 인코더 + 프로젝터 + 언어 디코더 결합 (추론 전용).

  - forward(**batch) → {"logits"}
  - generate(...)    : KV 캐시 기반 생성(새 토큰만 반환)
  - tie_weights / get_input_embeddings

이미지 병합: <|image|> 플레이스홀더 위치의 텍스트 임베딩을 프로젝션된 이미지 임베딩으로
평탄 대입한다(배치 전체에서 placeholder 수 == 총 타일수 × mp_image_token_length).

손실 계산·활성화 체크포인팅·파라미터 그룹 등 학습 경로는 학습 저장소 Pierrot-VLM-Lab 쪽에 있다.

텐서 차원 표기:
    B  = 배치, T = 시퀀스 길이(패딩 포함), D = lm_hidden_dim, V = lm_vocab_size
    n_img = 배치 내 총 이미지 타일 수, N' = mp_image_token_length(타일당 이미지 토큰 수)
    입력 batch: input_ids(B,T) / attention_mask(B,T 또는 None) /
                images = 샘플별 타일텐서 리스트 [ (타일수, 3, p, p), ... ]
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import VLMConfig
from .language_model import LanguageModel
from .modality_projector import ModalityProjector
from .vision_transformer import ViT


# ------------------------------------------------------------------ #
# top-k / nucleus(top-p) 필터링: 확률 낮은 토큰을 -inf 로 눌러 샘플링 후보를 줄인다.
# ------------------------------------------------------------------ #
def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0,
                          filter_value: float = -float("Inf")) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(remove, filter_value)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # 표준 nucleus: 누적확률이 top_p 를 '처음 초과하게 만든' 경계 토큰은 남긴다.
        # (remove mask 를 오른쪽으로 한 칸 밀어 경계 토큰을 보존 — 원본 nanoVLM 은 이걸
        #  일찍 제거해 다양성이 다소 작았다.)
        sorted_remove = cum > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = sorted_remove.scatter(1, sorted_idx, sorted_remove)
        logits = logits.masked_fill(remove, filter_value)
    return logits


class VisionLanguageModel(nn.Module):
    """SigLIP ViT + 픽셀셔플 프로젝터 + SmolLM2 디코더 결합 VLM."""

    # ------------------------------------------------------------------ #
    # load_backbone=True 면 공개 SigLIP/SmolLM2 백본에서 시작(from_pretrained),
    # 아니면 랜덤 초기화(공개 VLM 체크포인트로 덮어쓰거나 완전 스크래치용).
    # 프로젝터(MP)는 항상 랜덤. config 는 dataclass 로 보관(엔진 저장용).
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: VLMConfig, load_backbone: bool = True):
        super().__init__()
        self.cfg    = cfg
        self.config = cfg          # 엔진 save_pretrained 가 asdict(config) 로 config.json 저장
        if load_backbone:
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder        = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder        = LanguageModel(cfg)
        self.MP            = ModalityProjector(cfg)
        self.load_backbone = load_backbone

    # ------------------------------------------------------------------ #
    # head.weight 를 token_embedding 과 공유(weight tying).
    # ------------------------------------------------------------------ #
    def tie_weights(self) -> None:
        if self.cfg.lm_tie_weights:
            self.decoder.head.weight = self.decoder.token_embedding.weight

    # ------------------------------------------------------------------ #
    # 텍스트 임베딩 테이블(이미지 병합·tie 기준).
    # ------------------------------------------------------------------ #
    def get_input_embeddings(self) -> nn.Embedding:
        return self.decoder.token_embedding

    # ------------------------------------------------------------------ #
    # <|image|> 자리의 텍스트 임베딩을 프로젝션된 이미지 임베딩으로 평탄 대입한다.
    # 배치 전체 placeholder 수 == image_embd 총 토큰 수여야 한다(불일치 시 오류).
    # ------------------------------------------------------------------ #
    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        # token_embd:(B,T,D), image_embd:(n_img, N', D). <|image|> 자리만 교체.
        updated = token_embd.clone()
        mask    = (input_ids == self.image_token_id)                 # (B, T) True=이미지 자리
        # (n_img, N', D) → (n_img·N', D) 로 펼쳐 row-major 순서대로 True 위치에 채움
        updated[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated.dtype)
        return updated                                              # (B, T, D)

    # ------------------------------------------------------------------ #
    # 다양한 이미지 입력(타일 텐서 / 텐서 리스트 / 리스트의 리스트)을 하나의
    # (총타일, 3, p, p) 텐서로 평탄화하고 device·dtype 을 맞춘다. 없으면 None.
    # ------------------------------------------------------------------ #
    def _process_images(self, images, device, dtype):
        if images is None:
            return None
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sub in images for img in sub]
            if not images:
                return None
            images = torch.cat(images, dim=0)
        return images.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    # <|image|> 토큰 id (토크나이저 없이도 병합이 되도록 config/프로세서가 주입).
    # weights.load_pretrained 에서 set_image_token_id 로 설정된다.
    # ------------------------------------------------------------------ #
    @property
    def image_token_id(self) -> int:
        return self._image_token_id

    def set_image_token_id(self, token_id: int) -> None:
        self._image_token_id = int(token_id)

    # ------------------------------------------------------------------ #
    # 순전파.
    #   ① 텍스트: input_ids → 임베딩
    #   ② 이미지: 타일 → ViT → 프로젝터 → <|image|> 자리에 병합
    #   ③ 디코더(임베딩 모드) → hidden → head → 로짓
    # ------------------------------------------------------------------ #
    def forward(self, input_ids, images=None, attention_mask=None, **kwargs) -> Dict[str, torch.Tensor]:
        # ① 텍스트: input_ids(B,T) → 임베딩 (B,T,D). <|image|> 자리는 아래서 교체.
        token_embd    = self.decoder.token_embedding(input_ids)          # (B, T, D)
        images_tensor = self._process_images(images, input_ids.device, token_embd.dtype)  # (n_img,3,p,p) 또는 None
        if images_tensor is not None:
            # ② 이미지: 타일 → ViT (n_img, N, Dv) → 프로젝터 (n_img, N', D) → <|image|> 자리에 병합
            image_embd = self.MP(self.vision_encoder(images_tensor))     # (n_img, N', D)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)  # (B, T, D)

        # ③ 디코더(임베딩 모드) → hidden → head → 로짓
        hidden, _ = self.decoder(token_embd, attention_mask=attention_mask)  # (B, T, D)
        logits    = self.decoder.head(hidden)                                # (B, T, V)
        return {"logits": logits}

    # ------------------------------------------------------------------ #
    # KV 캐시 기반 autoregressive 생성. 새로 생성된 토큰 id (B, ≤max_new_tokens)만 반환.
    #   - 프리필: 이미지 병합된 프롬프트를 한 번에 처리해 캐시를 채우고 첫 로짓 획득
    #   - 디코드: 새 토큰 1개만 임베딩해 캐시 참조, greedy 또는 top-k/p 샘플링
    #   - 각 행의 첫 EOS 이후는 모두 EOS 로 덮어 정리
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(self, input_ids, images=None, attention_mask=None, max_new_tokens: int = 5,
                 top_k: int = 50, top_p: float = 0.9, temperature: float = 0.5,
                 greedy: bool = False, eos_token_id: Optional[int] = None):
        # 생성 인자 검증(잘못된 값은 조용히 이상 동작 대신 즉시 오류).
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens 는 0 이상이어야 합니다: {max_new_tokens}")
        if not greedy:                                         # 샘플링 모드에서만 의미 있는 인자들
            if temperature <= 0:
                raise ValueError(f"샘플링 temperature 는 0 보다 커야 합니다(0 은 greedy 사용): {temperature}")
            if not (0.0 < top_p <= 1.0):
                raise ValueError(f"top_p 는 (0, 1] 범위여야 합니다: {top_p}")
            if top_k < 0:
                raise ValueError(f"top_k 는 0 이상이어야 합니다(0=미사용): {top_k}")

        # 프롬프트 임베딩 준비(이미지 병합) — forward 와 동일. (B, T, D)
        token_embd    = self.decoder.token_embedding(input_ids)
        images_tensor = self._process_images(images, input_ids.device, token_embd.dtype)
        if images_tensor is not None:
            image_embd = self.MP(self.vision_encoder(images_tensor))         # (n_img, N', D)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        cur_len = token_embd.size(1)                        # 현재 시퀀스 길이 T
        B       = input_ids.size(0)

        eos = eos_token_id if eos_token_id is not None else getattr(self, "_eos_token_id", None)

        # 프리필: 프롬프트 전체를 한 번에 처리해 KV 캐시를 채우고, 마지막 위치 로짓만 취함.
        prefill, kv_cache = self.decoder(token_embd, attention_mask=attention_mask, kv_cache=None, start_pos=0)  # (B, T, D)
        logits = self.decoder.head(prefill[:, -1, :])       # 마지막 위치 → (B, V)

        new_ids  = []
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        for step in range(max_new_tokens):
            if greedy:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                filtered = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs    = torch.softmax(filtered / temperature, dim=-1)
                nxt      = torch.multinomial(probs, num_samples=1)
            new_ids.append(nxt)

            # 조기 종료: 모든 시퀀스가 EOS 를 냈으면 남은 디코드를 낭비하지 않는다.
            if eos is not None:
                finished = finished | (nxt.squeeze(1) == eos)
                if bool(finished.all()):
                    break
            if step == max_new_tokens - 1:
                break                                  # 마지막 토큰이면 다음 로짓 계산 불필요

            # 디코드: 새 토큰 1개만 임베딩해 캐시를 참조(1스텝). emb:(B,1,D)
            emb       = self.decoder.token_embedding(nxt)
            start_pos = cur_len                             # 새 토큰의 위치(캐시 길이 뒤)
            cur_len  += 1
            if attention_mask is not None:
                attention_mask = torch.cat(                 # 새 토큰 자리(1)를 마스크에 추가 → (B, T+step+1)
                    [attention_mask, torch.ones((B, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
            out, kv_cache = self.decoder(emb, attention_mask=attention_mask, kv_cache=kv_cache, start_pos=start_pos)  # (B, 1, D)
            logits = self.decoder.head(out[:, -1, :])       # (B, V)

        if not new_ids:
            return torch.empty((B, 0), dtype=torch.long, device=input_ids.device)
        gen = torch.cat(new_ids, dim=1)

        # 첫 EOS 이후 토큰을 모두 EOS 로 정리(배치 디코드 편의).
        if eos is not None and gen.numel() > 0:
            T   = gen.size(1)
            col = torch.arange(T, device=gen.device)
            eos_mask   = (gen == eos)
            first_eos  = torch.where(eos_mask, col.unsqueeze(0).expand_as(gen), T + 1).min(dim=1).values.clamp(max=T)
            replace    = col.unsqueeze(0).expand_as(gen) > first_eos.unsqueeze(1)
            gen[replace] = eos
        return gen

    # ------------------------------------------------------------------ #
    # 공개(HF Hub)/로컬 VLM 체크포인트에서 로드(config.json + model.safetensors).
    # (Pierrot weights.load_pretrained 도 이 규약을 재사용한다.)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, repo_id_or_path: str, revision: Optional[str] = None) -> "VisionLanguageModel":
        from safetensors.torch import load_model

        if os.path.exists(repo_id_or_path):
            config_path  = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")
        else:
            from huggingface_hub import hf_hub_download
            config_path  = hf_hub_download(repo_id=repo_id_or_path, filename="config.json", revision=revision)
            weights_path = hf_hub_download(repo_id=repo_id_or_path, filename="model.safetensors", revision=revision)

        with open(config_path) as f:
            cfg = VLMConfig(**json.load(f))
        model = cls(cfg, load_backbone=False)
        load_model(model, weights_path)
        return model

    # ------------------------------------------------------------------ #
    # config.json + model.safetensors 로 저장(공개 nanoVLM 포맷과 호환).
    # ------------------------------------------------------------------ #
    def save_pretrained(self, save_directory: str) -> None:
        from safetensors.torch import save_model

        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=4)
        save_model(self, os.path.join(save_directory, "model.safetensors"))
