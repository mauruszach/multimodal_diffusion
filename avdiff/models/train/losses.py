#!/usr/bin/env python3
"""
losses.py — noise MSE for targets and optional alignment loss.
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def mse_targets_only(
    eps_hat: Dict[str, torch.Tensor],
    eps_true: Dict[str, torch.Tensor],
    target: str,  # "video" or "audio"
) -> torch.Tensor:
    """
    eps_hat/true: dicts with keys "video" and/or "audio"
      video: [B, Nv, Dv]
      audio: [B, Na, Da]
    """
    if target == "video":
        return F.mse_loss(eps_hat["video"], eps_true["video"])
    elif target == "audio":
        return F.mse_loss(eps_hat["audio"], eps_true["audio"])
    else:
        raise ValueError("target must be 'video' or 'audio'")


def alignment_loss(
    h_video: Optional[torch.Tensor],
    h_audio: Optional[torch.Tensor],
    weight: float = 0.0,
    method: str = "cosine",
) -> torch.Tensor:
    """
    Weak alignment between contextualized features (post-core).
    h_video: [B, Nv, d], h_audio: [B, Na, d]
    Returns weight * loss (0 if inputs are None or weight=0).
    """
    if weight <= 0.0 or h_video is None or h_audio is None:
        return torch.tensor(0.0, device=h_video.device if h_video is not None else h_audio.device)

    # simple mean-pool over tokens
    v = h_video.mean(dim=1)
    a = h_audio.mean(dim=1)

    if method == "cosine":
        v = F.normalize(v, dim=-1)
        a = F.normalize(a, dim=-1)
        cos = (v * a).sum(dim=-1)  # [B]
        loss = 1.0 - cos.mean()
    elif method == "l2":
        loss = F.mse_loss(v, a)
    else:
        raise ValueError("Unknown alignment method")

    return weight * loss
