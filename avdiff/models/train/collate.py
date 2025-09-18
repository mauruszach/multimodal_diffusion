#!/usr/bin/env python3
"""
collate.py — build batches, masks, and target selection (V→A or A→V).
Assumes dataset items shaped like:
{
  "video": FloatTensor [3, T, H, W]   (in [0,1]), or None if missing
  "audio": FloatTensor [1, L]         (mono, in [-1,1]), or None if missing
  "meta":  dict(...)                  (optional)
}
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple

import torch


def _pad_video(v: torch.Tensor, T: int) -> torch.Tensor:
    # v: [3,T0,H,W] -> pad to T on time dimension by repeating last frame
    if v is None:
        return None
    t0 = v.size(1)
    if t0 == T:
        return v
    if t0 > T:
        return v[:, :T]
    last = v[:, -1:].expand(-1, T - t0, -1, -1)
    return torch.cat([v, last], dim=1)


def _pad_audio(a: torch.Tensor, L: int) -> torch.Tensor:
    # a: [1,L0] -> pad with zeros on the right
    if a is None:
        return None
    l0 = a.size(-1)
    if l0 == L:
        return a
    if l0 > L:
        return a[..., :L]
    pad = torch.zeros(1, L - l0, dtype=a.dtype, device=a.device)
    return torch.cat([a, pad], dim=-1)


def collate_batch(
    items: List[Dict[str, Any]],
    T_target: int,
    L_target: int,
    pick_target: str,  # "video" or "audio" for this batch
) -> Dict[str, Any]:
    """
    Returns:
      {
        "video": FloatTensor [B,3,T,H,W]  (may be None if all missing),
        "audio": FloatTensor [B,1,L]      (may be None if all missing),
        "has_video": BoolTensor [B],
        "has_audio": BoolTensor [B],
        "target": str ("video" or "audio")
      }
    """
    B = len(items)
    vids = []
    auds = []
    has_v = []
    has_a = []

    H = W = None
    for it in items:
        v = it.get("video", None)
        a = it.get("audio", None)
        if v is not None:
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            if H is None:
                H, W = v.size(-2), v.size(-1)
        if a is not None and not isinstance(a, torch.Tensor):
            a = torch.as_tensor(a)

        vids.append(v)
        auds.append(a)
        has_v.append(v is not None)
        has_a.append(a is not None)

    # If some items miss a modality, fill with zeros to keep batch rectangular
    # (the trainer will ignore missing modality for encoding)
    v_batch = None
    if any(has_v):
        v_filled = []
        for v in vids:
            if v is None:
                v = torch.zeros(3, T_target, H, W)
            else:
                v = _pad_video(v, T_target)
            v_filled.append(v)
        v_batch = torch.stack(v_filled, dim=0)  # [B,3,T,H,W]

    a_batch = None
    if any(has_a):
        a_filled = []
        for a in auds:
            if a is None:
                a = torch.zeros(1, L_target)
            else:
                a = _pad_audio(a, L_target)
            a_filled.append(a)
        a_batch = torch.stack(a_filled, dim=0)  # [B,1,L]

    return {
        "video": v_batch,
        "audio": a_batch,
        "has_video": torch.tensor(has_v, dtype=torch.bool),
        "has_audio": torch.tensor(has_a, dtype=torch.bool),
        "target": pick_target,
        "meta": [it.get("meta", {}) for it in items],
    }
