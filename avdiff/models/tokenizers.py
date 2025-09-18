#!/usr/bin/env python3
"""
tokenizers.py — lightweight tokenizers for video/audio latents.

Video:
  - tube patching over latent [B, C, T, H, W] -> tokens [B, N, C*t*h*w]
Audio:
  - 1D chunking over latent [B, C, L] -> tokens [B, N, C*chunk_len]

Also includes the inverse mappings used to fold token-aligned tensors back to
latent layouts (e.g., to aggregate predicted ε into latent-shaped ε).

These are thin wrappers around avdiff.utils.ops with nice classes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch

from avdiff.utils import ops


# ---------------- Video ----------------

@dataclass
class VideoTokenizerCfg:
    tube_t: int = 2
    tube_h: int = 4
    tube_w: int = 4

class VideoTokenizer(torch.nn.Module):
    """
    Converts video latents <-> tokens using tube patching.

    encode(z):  [B, C, T, H, W] -> [B, N, C*t*h*w]
    decode(tok, C,T,H,W): tokens back to latent layout
    """
    def __init__(self, cfg: VideoTokenizerCfg | None = None):
        super().__init__()
        self.cfg = cfg or VideoTokenizerCfg()

    @property
    def tube(self) -> Tuple[int, int, int]:
        return self.cfg.tube_t, self.cfg.tube_h, self.cfg.tube_w

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        t, h, w = self.tube
        return ops.tube_patch_video(z, t=t, h=h, w=w)

    def decode(self, tokens: torch.Tensor, C: int, T: int, H: int, W: int) -> torch.Tensor:
        t, h, w = self.tube
        return ops.tube_unpatch_video(tokens, C=C, T=T, H=H, W=W, t=t, h=h, w=w)

    def token_dim(self, C_latent: int) -> int:
        t, h, w = self.tube
        return C_latent * t * h * w


# ---------------- Audio ----------------

@dataclass
class AudioTokenizerCfg:
    chunk_len: int = 4
    stride: int = 4

class AudioTokenizer(torch.nn.Module):
    """
    Converts audio latents <-> tokens using 1D chunking.

    encode(z):  [B, C, L] -> [B, N, C*chunk_len]
    decode(tok, C, L): tokens back to [B, C, L] via overlap-add with rectangular window.
    """
    def __init__(self, cfg: AudioTokenizerCfg | None = None):
        super().__init__()
        self.cfg = cfg or AudioTokenizerCfg()

    @property
    def chunk(self) -> Tuple[int, int]:
        return self.cfg.chunk_len, self.cfg.stride

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        length, stride = self.chunk
        windows = ops.chunk_1d(z, length=length, stride=stride, dim=-1)  # [B, C, N, length]
        B, C, N, L = windows.shape
        tokens = windows.permute(0, 2, 1, 3).contiguous().view(B, N, C * L)  # [B, N, C*len]
        return tokens

    def decode(self, tokens: torch.Tensor, C: int, L_total: int) -> torch.Tensor:
        """
        Inverse fold using overlap-add. Expects tokens of shape [B, N, C*chunk_len].
        """
        length, stride = self.chunk
        B, N, D = tokens.shape
        assert D == C * length, f"token dim {D} != C*chunk_len {C*length}"
        windows = tokens.view(B, N, C, length).permute(0, 2, 1, 3).contiguous()  # [B, C, N, length]
        outs = []
        for b in range(B):
            chs = []
            for c in range(C):
                # ops.overlap_add_1d expects [..., N, length]; give [1, N, length]
                y = ops.overlap_add_1d(windows[b, c].unsqueeze(0), stride=stride, length=length, apply_hann=False)
                chs.append(y)  # [1, L]
            z = torch.stack(chs, dim=1)  # [1, C, L]
            outs.append(z)
        out = torch.cat(outs, dim=0)  # [B, C, L]
        # trim/pad to L_total
        if out.size(-1) > L_total:
            out = out[..., :L_total]
        elif out.size(-1) < L_total:
            pad = torch.zeros(out.size(0), out.size(1), L_total - out.size(-1), device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=-1)
        return out

    def token_dim(self, C_latent: int) -> int:
        return C_latent * self.cfg.chunk_len
