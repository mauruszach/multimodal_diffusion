#!/usr/bin/env python3
"""
mmdt.py — Multimodal Diffusion Transformer (shared denoiser core).

A clean, DiT-style transformer encoder that operates on the concatenated
token sequence (video tokens followed by audio tokens). It’s modality-agnostic;
all modality bias comes from the adapters/embeddings you feed it.

Config (matches earlier YAML):
  d_model: 1024
  n_layers: 16
  n_heads: 16
  mlp_ratio: 4.0
  dropout: 0.1
  attn_dropout: 0.0
  norm: "rmsnorm" | "layernorm"
  rope: false
  token_dropout: 0.0
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------- Norms --------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d]
        norm_x = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return self.scale * x / (norm_x + self.eps)


def build_norm(kind: str, d: int) -> nn.Module:
    return RMSNorm(d) if kind.lower() == "rmsnorm" else nn.LayerNorm(d)


# -------------- Attention (MHA wrapper) --------------

class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0, resid_dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                         dropout=attn_dropout, batch_first=True)
        self.drop = nn.Dropout(resid_dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        # nn.MultiheadAttention uses [B, N, d] with batch_first=True
        y, _ = self.mha(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return self.drop(y)


# -------------- MLP --------------

class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -------------- Block --------------

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float, attn_dropout: float, norm: str):
        super().__init__()
        self.norm1 = build_norm(norm, d_model)
        self.attn = MHA(d_model, n_heads, attn_dropout=attn_dropout, resid_dropout=dropout)
        self.norm2 = build_norm(norm, d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# -------------- Core --------------

@dataclass
class MMDiTCfg:
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    norm: str = "rmsnorm"
    rope: bool = False
    token_dropout: float = 0.0  # stochastic token drop during training

class MMDiT(nn.Module):
    """
    Multimodal Diffusion Transformer — a stack of self-attention blocks over the
    concatenated token sequence. It returns contextualized tokens [B, N, d].

    Inputs:
      x: [B, N, d]
      key_padding_mask (optional): [B, N] True where tokens are PAD (mask out)
    """
    def __init__(self, d_model=1024, n_layers=16, n_heads=16, mlp_ratio=4.0,
                 dropout=0.1, attn_dropout=0.0, norm="rmsnorm", rope=False, token_dropout=0.0):
        super().__init__()
        self.cfg = MMDiTCfg(d_model, n_layers, n_heads, mlp_ratio, dropout, attn_dropout, norm, rope, token_dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, mlp_ratio, dropout, attn_dropout, norm) for _ in range(n_layers)
        ])
        self.final_norm = build_norm(norm, d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, N, d]
        key_padding_mask: [B, N] with True for positions that are padding
        """
        if self.training and self.cfg.token_dropout > 0.0:
            # stochastic token dropout: randomly zero some tokens (same mask across dim)
            keep = torch.rand(x.shape[0], x.shape[1], device=x.device) > self.cfg.token_dropout
            keep = keep.unsqueeze(-1).to(x.dtype)
            x = x * keep

        # broadcast mask to nn.MultiheadAttention format (True = ignore)
        for blk in self.blocks:
            x = blk(x, attn_mask=None, key_padding_mask=key_padding_mask)

        return self.final_norm(x)
