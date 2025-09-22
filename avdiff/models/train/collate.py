#!/usr/bin/env python3
"""
collate.py — build batches, masks, and target selection (V→A or A→V).

Expected dataset item:
{
  "video": FloatTensor [3, T, H, W]   (in [0,1]) or None
  "audio": FloatTensor [1, L]         (mono in [-1,1]) or None
  "meta":  dict(...)                  (optional)
}

This collate:
- Pads/crops video to exactly T_target frames (repeat last frame if short)
- Pads/crops audio to exactly L_target samples (zero-pad if short)
- Returns missing modalities as zero tensors; you can ignore them in the trainer
- Accepts `pick_target` either as a callable () -> {"video"}|{"audio"} or a string
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set

import torch


def _as_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def _pad_video(v: torch.Tensor, T: int) -> torch.Tensor:
    """
    v: [3, T0, H, W] -> pad/crop to T on time dimension.
    If T0 < T, repeat the last frame; if T0 > T, truncate.
    """
    t0 = v.size(1)
    if t0 == T:
        return v
    if t0 > T:
        return v[:, :T]
    # repeat last frame
    last = v[:, -1:].expand(-1, T - t0, -1, -1)
    return torch.cat([v, last], dim=1)


def _pad_audio(a: torch.Tensor, L: int) -> torch.Tensor:
    """
    a: [1, L0] -> pad/crop to L on last dimension.
    If L0 < L, right-pad zeros; if L0 > L, truncate.
    """
    l0 = a.size(-1)
    if l0 == L:
        return a
    if l0 > L:
        return a[..., :L]
    pad = torch.zeros(1, L - l0, dtype=a.dtype, device=a.device)
    return torch.cat([a, pad], dim=-1)


def _decide_target(
    pick_target: Optional[Union[str, Set[str], Callable[[], Union[str, Set[str]]]]],
    has_video: bool,
    has_audio: bool,
) -> Set[str]:
    """
    Normalize target selection to a set {"video"} or {"audio"}.
    Fallback to the modality that exists if only one is present.
    """
    # Resolve callable or literal
    target: Optional[Union[str, Set[str]]] = None
    if callable(pick_target):
        target = pick_target()
    else:
        target = pick_target

    # Normalize to set[str]
    if isinstance(target, str):
        target_set = {target}
    elif isinstance(target, set):
        target_set = target
    elif target is None:
        # default 50/50 if both exist, else whichever exists
        if has_video and has_audio:
            target_set = {"audio"} if torch.rand(()) < 0.5 else {"video"}
        elif has_video:
            target_set = {"video"}
        elif has_audio:
            target_set = {"audio"}
        else:
            target_set = {"audio"}  # arbitrary; batch is empty anyway
    else:
        raise ValueError(f"Unsupported target type: {type(target)}")

    # If user requested a target that doesn't exist in this batch, fallback
    if "video" in target_set and not has_video and has_audio:
        return {"audio"}
    if "audio" in target_set and not has_audio and has_video:
        return {"video"}
    return target_set


def collate_batch(
    items: List[Dict[str, Any]],
    T_target: int,
    L_target: int,
    pick_target: Optional[Union[str, Set[str], Callable[[], Union[str, Set[str]]]]] = None,
) -> Dict[str, Any]:
    """
    Build a rectangular batch with padding and choose a training target.

    Returns:
      {
        "video": FloatTensor [B, 3, T_target, H, W] or None if all missing
        "audio": FloatTensor [B, 1, L_target]       or None if all missing
        "has_video": BoolTensor [B]
        "has_audio": BoolTensor [B]
        "target": set[str]   # {"video"} or {"audio"}
        "meta":   list[dict]
      }
    """
    B = len(items)
    vids: List[Optional[torch.Tensor]] = []
    auds: List[Optional[torch.Tensor]] = []
    metas: List[Dict[str, Any]] = []
    has_v_flags: List[bool] = []
    has_a_flags: List[bool] = []

    H: Optional[int] = None
    W: Optional[int] = None

    # Gather, coerce to tensor, infer H/W
    for it in items:
        v = it.get("video", None)
        a = it.get("audio", None)

        if v is not None:
            v = _as_tensor(v)  # expect [3, T, H, W]
            if v.dim() != 4 or v.size(0) != 3:
                raise ValueError(f"video must be [3, T, H, W]; got shape {tuple(v.shape)}")
            if H is None:
                H, W = int(v.size(-2)), int(v.size(-1))

        if a is not None:
            a = _as_tensor(a)  # expect [1, L]
            if a.dim() != 2 or a.size(0) != 1:
                raise ValueError(f"audio must be [1, L]; got shape {tuple(a.shape)}")

        vids.append(v)
        auds.append(a)
        metas.append(it.get("meta", {}))
        has_v_flags.append(v is not None)
        has_a_flags.append(a is not None)

    has_any_video = any(has_v_flags)
    has_any_audio = any(has_a_flags)

    # If no sample provides video but we need shapes, set H/W to a default
    if not has_any_video and (H is None or W is None):
        # Default to 128x128; change if your pipeline guarantees other sizes
        H, W = 128, 128

    # Fill missing with zeros and pad/crop to targets
    v_batch: Optional[torch.Tensor] = None
    if has_any_video:
        v_filled: List[torch.Tensor] = []
        for v in vids:
            if v is None:
                v = torch.zeros(3, T_target, H, W, dtype=torch.float32)
            else:
                # If H/W vary (shouldn't in curated data), center-crop or pad to (H,W)
                h, w = v.size(-2), v.size(-1)
                if (h, w) != (H, W):
                    # basic center-crop or pad
                    dh = max(0, H - h)
                    dw = max(0, W - w)
                    if dh > 0 or dw > 0:
                        # pad to the right/bottom
                        v = torch.nn.functional.pad(v, (0, dw, 0, dh))
                        h, w = v.size(-2), v.size(-1)
                    if h > H or w > W:
                        v = v[..., :H, :W]
                v = _pad_video(v, T_target)
            v_filled.append(v)
        v_batch = torch.stack(v_filled, dim=0)  # [B, 3, T, H, W]

    a_batch: Optional[torch.Tensor] = None
    if has_any_audio:
        a_filled: List[torch.Tensor] = []
        for a in auds:
            if a is None:
                a = torch.zeros(1, L_target, dtype=torch.float32)
            else:
                a = _pad_audio(a, L_target)
            a_filled.append(a)
        a_batch = torch.stack(a_filled, dim=0)  # [B, 1, L]

    # Decide training target (as a set)
    target_set = _decide_target(
        pick_target=pick_target,
        has_video=has_any_video,
        has_audio=has_any_audio,
    )

    return {
        "video": v_batch,
        "audio": a_batch,
        "has_video": torch.tensor(has_v_flags, dtype=torch.bool),
        "has_audio": torch.tensor(has_a_flags, dtype=torch.bool),
        "target": target_set,          # {"video"} or {"audio"}
        "meta": metas,
    }


__all__ = ["collate_batch"]
