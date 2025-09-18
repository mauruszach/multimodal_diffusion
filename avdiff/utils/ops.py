#!/usr/bin/env python3
"""
ops.py — small tensor ops and chunk/patch helpers.
"""

from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn.functional as F


# ---------------------------
# 1D chunking (audio / smell)
# ---------------------------

def chunk_1d(x: torch.Tensor, length: int, stride: int, dim: int = -1) -> torch.Tensor:
    """
    Create non-overlapping (or strided) chunks along a time dimension.

    Args:
      x: tensor [..., L]
      length: chunk length
      stride: step between chunks
      dim: dimension to chunk (default: last)

    Returns:
      windows: [..., N, length] where N = floor((L - length)/stride) + 1
    """
    if dim != x.dim() - 1:
        x = x.transpose(dim, -1)
        transposed = True
    else:
        transposed = False

    L = x.size(-1)
    if length <= 0 or stride <= 0 or L < length:
        # Return a single (possibly shorter) window
        out = x[..., :max(0, min(L, length))].unsqueeze(-2)
    else:
        out = x.unfold(dimension=-1, size=length, step=stride)  # [..., N, length]

    if transposed:
        out = out.transpose(dim, -1)  # put back the dim order
    return out


def overlap_add_1d(windows: torch.Tensor,
                   stride: int,
                   length: Optional[int] = None,
                   dim_windows: int = -2,
                   apply_hann: bool = False) -> torch.Tensor:
    """
    Reconstruct 1D signal via overlap-add from windows.

    Args:
      windows: [..., N, length] (N windows on dim_windows)
      stride: hop between windows
      length: optional window length override
      dim_windows: which dim indexes the window count N
      apply_hann: if True, divide by Hann window overlap normalization

    Returns:
      y: reconstructed signal of shape [..., L]
    """
    if length is None:
        length = windows.size(-1)

    # Move windows dim to -2 for consistent math
    if dim_windows != windows.dim() - 2:
        perm = list(range(windows.dim()))
        perm.pop(dim_windows)
        perm.insert(windows.dim() - 2, dim_windows)
        windows = windows.permute(*perm)

    *prefix, N, W = windows.shape
    L_out = (N - 1) * stride + W
    y = windows.new_zeros(*prefix, L_out)
    norm = windows.new_zeros(L_out)

    if apply_hann:
        win = torch.hann_window(W, device=windows.device, dtype=windows.dtype)
    else:
        win = torch.ones(W, device=windows.device, dtype=windows.dtype)

    for i in range(N):
        start = i * stride
        y[..., start:start + W] += windows[..., i, :] * win
        norm[start:start + W] += win

    norm = torch.clamp(norm, min=1e-8)
    y = y / norm
    return y


# ---------------------------
# Video tube patching
# ---------------------------

def tube_patch_video(z: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
    """
    Convert video latent [B, C, T, H, W] into tube tokens [B, N, C*t*h*w].

    Args:
      z: [B, C, T, H, W]
      t,h,w: tube sizes (must divide T,H,W)

    Returns:
      tokens: [B, N, C*t*h*w] where
        N = (T/t) * (H/h) * (W/w)
    """
    B, C, T, H, W = z.shape
    assert T % t == 0 and H % h == 0 and W % w == 0, "tube sizes must divide latent dims"
    z = z.view(B, C, T // t, t, H // h, h, W // w, w)
    # [B, C, T', t, H', h, W', w] -> [B, T', H', W', C, t, h, w]
    z = z.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    N = (T // t) * (H // h) * (W // w)
    tokens = z.view(B, N, C * t * h * w)
    return tokens


def tube_unpatch_video(tokens: torch.Tensor,
                       C: int, T: int, H: int, W: int,
                       t: int, h: int, w: int) -> torch.Tensor:
    """
    Inverse of tube_patch_video.

    Args:
      tokens: [B, N, C*t*h*w]
      C,T,H,W: target latent shape
      t,h,w: tube sizes

    Returns:
      z: [B, C, T, H, W]
    """
    B, N, D = tokens.shape
    assert D == C * t * h * w, "token width mismatch"
    Tt, Hh, Ww = T // t, H // h, W // w
    assert N == Tt * Hh * Ww, "token count mismatch"
    z = tokens.view(B, Tt, Hh, Ww, C, t, h, w)
    # [B, T', H', W', C, t, h, w] -> [B, C, T', t, H', h, W', w]
    z = z.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    z = z.view(B, C, T, H, W)
    return z


# ---------------------------
# Misc small helpers
# ---------------------------

def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -1, value: float = 0.0) -> Tuple[torch.Tensor, int]:
    """
    Right-pad x on `dim` to be a multiple of `multiple`. Returns (padded, pad_amount).
    """
    size = x.size(dim)
    pad_amt = (multiple - size % multiple) % multiple
    if pad_amt == 0:
        return x, 0
    pad_shape = [0, 0] * x.dim()
    pad_shape[-1 - 2 * (x.dim() - 1 - dim)] = pad_amt  # map to F.pad's reverse-order spec
    return F.pad(x, pad=pad_shape, value=value), pad_amt
