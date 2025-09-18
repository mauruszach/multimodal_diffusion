#!/usr/bin/env python3
"""
adapters.py — projections to token width d and lightweight embeddings.

Pieces:
  - LinearAdapter: project per-token raw dim -> width d
  - ModalityEmbedding: learned {video,audio} embedding added per token
  - PositionalEmbedding3D: learned/sinusoidal over (T',H',W'), summed and gathered per token
  - PositionalEmbedding1D: learned/sinusoidal over N (audio token index)
  - TimestepEmbedder: sinusoidal or small MLP from scalar t -> [B, d], added or concatenated

These are modular; use the ones you need in training/inference.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import math
import torch
import torch.nn as nn

from avdiff.utils import schedule_utils as su


# ---------------- Basic adapters ----------------

class LinearAdapter(nn.Module):
    """Linear projection of per-token features to width d."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, d_in]
        return self.proj(x)


class ModalityEmbedding(nn.Module):
    """Adds a learned embedding per token depending on modality."""
    def __init__(self, d: int, modalities=("video", "audio")):
        super().__init__()
        self.lookup = nn.Embedding(len(modalities), d)
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.02)
        self.index = {m: i for i, m in enumerate(modalities)}

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        emb = self.lookup.weight[self.index[modality]].view(1, 1, -1)
        return x + emb.to(x.device, x.dtype)


# ---------------- Positional embeddings ----------------

def _sinusoid_position(n: int, d: int, device, dtype):
    pe = torch.zeros(n, d, device=device, dtype=dtype)
    position = torch.arange(0, n, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [n, d]

class PositionalEmbedding1D(nn.Module):
    """
    1D positional embeddings for audio tokens.
    mode: 'learned' or 'sin'
    """
    def __init__(self, d: int, max_len: int = 4096, mode: Literal["learned", "sin"] = "learned"):
        super().__init__()
        self.d = d
        self.mode = mode
        self.max_len = max_len
        if mode == "learned":
            self.table = nn.Embedding(max_len, d)
            nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, B: int, N: int, device, dtype) -> torch.Tensor:
        if self.mode == "learned":
            idx = torch.arange(N, device=device)
            pe = self.table(idx)  # [N, d]
        else:
            pe = _sinusoid_position(N, self.d, device=device, dtype=dtype)
        return pe.unsqueeze(0).expand(B, N, -1)  # [B, N, d]


class PositionalEmbedding3D(nn.Module):
    """
    3D positional embeddings for video tokens at grid (T',H',W').
    Supports 'learned' (separate tables per axis summed) or 'sin'.
    """
    def __init__(self, d: int, max_T: int = 256, max_H: int = 256, max_W: int = 256,
                 mode: Literal["learned", "sin"] = "learned"):
        super().__init__()
        self.d = d
        self.mode = mode
        if mode == "learned":
            self.t_table = nn.Embedding(max_T, d)
            self.h_table = nn.Embedding(max_H, d)
            self.w_table = nn.Embedding(max_W, d)
            nn.init.normal_(self.t_table.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.h_table.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.w_table.weight, mean=0.0, std=0.02)

    def forward(self, B: int, Tt: int, Hh: int, Ww: int, device, dtype) -> torch.Tensor:
        """
        Returns per-token PE: [B, N, d] with N=Tt*Hh*Ww in raster order (t-major, then h,w).
        """
        N = Tt * Hh * Ww
        if self.mode == "learned":
            t_idx = torch.arange(Tt, device=device)
            h_idx = torch.arange(Hh, device=device)
            w_idx = torch.arange(Ww, device=device)
            # grid
            tt = self.t_table(t_idx)[:, None, None, :]  # [Tt,1,1,d]
            hh = self.h_table(h_idx)[None, :, None, :]  # [1,Hh,1,d]
            ww = self.w_table(w_idx)[None, None, :, :]  # [1,1,Ww,d]
            pe = tt + hh + ww                            # [Tt,Hh,Ww,d]
            pe = pe.view(N, self.d)                      # [N,d]
        else:
            # sinusoidal via flattened 3D indices (simple but effective)
            idx = torch.arange(N, device=device)
            pe = _sinusoid_position(N, self.d, device=device, dtype=dtype)  # [N,d]
            pe = pe.index_select(0, idx)

        return pe.unsqueeze(0).expand(B, -1, -1)  # [B, N, d]


# ---------------- Time embedding ----------------

@dataclass
class TimestepCfg:
    dim: int = 256
    mode: Literal["sin", "mlp"] = "sin"

class TimestepEmbedder(nn.Module):
    """
    Maps integer timesteps t[B] -> embedding [B, d_t]. Typically ADD to tokens.
    """
    def __init__(self, cfg: TimestepCfg):
        super().__init__()
        self.cfg = cfg
        if cfg.mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(cfg.dim, cfg.dim * 2),
                nn.SiLU(),
                nn.Linear(cfg.dim * 2, cfg.dim),
            )
        else:
            self.mlp = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # sinusoidal base (B, dim)
        base = su.timestep_embedding(t, dim=self.cfg.dim)
        if self.mlp is not None:
            base = self.mlp(base)
        return base  # [B, d_t]
