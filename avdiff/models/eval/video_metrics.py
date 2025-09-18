#!/usr/bin/env python3
"""
video_metrics.py — basic video metrics & hooks.

Provided:
  - PSNR (per-frame & mean)
  - SSIM (per-frame & mean) via scikit-image (if available)
  - LPIPS (mean) if `lpips` package is installed
  - Temporal flicker: mean |frame[t] - frame[t-1]| (no reference)

CLI examples:
  # Reference vs generated (PSNR/SSIM/LPIPS)
  python -m avdiff.eval.video_metrics --ref ref_frames_dir --est gen_frames_dir

  # Generated only (temporal flicker)
  python -m avdiff.eval.video_metrics --est gen_frames_dir
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# SSIM/PSNR
try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
except Exception:
    ssim_fn = None
    psnr_fn = None

# LPIPS (optional)
try:
    import torch
    import lpips as lpips_lib  # pip install lpips
except Exception:
    torch = None
    lpips_lib = None


def _list_frames(frames_dir: Path, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    files = []
    for e in exts:
        files.extend(sorted(frames_dir.glob(f"*{e}")))
    return files

def _read_frames(frames_dir: Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV required to read images.")
    paths = _list_frames(frames_dir)
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)  # [T,H,W,3]

def _to_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return x

def _psnr_ssim_pair(ref: np.ndarray, est: np.ndarray) -> Tuple[float, float]:
    """
    Compute PSNR/SSIM for a pair of frames in [0,1].
    """
    if psnr_fn is None or ssim_fn is None:
        return float("nan"), float("nan")
    psnr = float(psnr_fn(ref, est, data_range=1.0))
    try:
        ssim = float(ssim_fn(ref, est, data_range=1.0, channel_axis=2))
    except TypeError:
        # older skimage: use multichannel flag
        ssim = float(ssim_fn(ref, est, data_range=1.0, multichannel=True))
    return psnr, ssim

def _lpips_model(device: str = "cuda") -> Optional[any]:
    if lpips_lib is None or torch is None:
        return None
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = lpips_lib.LPIPS(net="alex").to(dev)
    model.eval()
    return model

def _lpips_pair(model, ref: np.ndarray, est: np.ndarray) -> float:
    """
    Compute LPIPS between two frames (expects [H,W,3] in [0,1]).
    """
    if model is None:
        return float("nan")
    # to [-1,1] torch CHW
    t_ref = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    t_est = torch.from_numpy(est).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    t_ref = t_ref.to(next(model.parameters()).device).float()
    t_est = t_est.to(next(model.parameters()).device).float()
    with torch.no_grad():
        d = model(t_ref, t_est)
    return float(d.squeeze().item())

def temporal_flicker(frames: np.ndarray) -> float:
    """
    Mean absolute difference between consecutive frames (in [0,1]).
    Higher = more flicker (worse temporal consistency).
    """
    x = _to_float01(frames)
    if x.shape[0] < 2:
        return 0.0
    diffs = np.abs(x[1:] - x[:-1]).mean(axis=(1, 2, 3))
    return float(diffs.mean())


# -------------- Public API --------------

def evaluate_video_pair(ref_dir: Path, est_dir: Path, lpips_device: str = "cuda") -> Dict[str, float]:
    ref = _to_float01(_read_frames(ref_dir))
    est = _to_float01(_read_frames(est_dir))
    T = min(ref.shape[0], est.shape[0])
    ref = ref[:T]; est = est[:T]

    psnrs, ssims = [], []
    for t in range(T):
        p, s = _psnr_ssim_pair(ref[t], est[t])
        psnrs.append(p); ssims.append(s)

    lpips_model = _lpips_model(lpips_device)
    lpips_vals = []
    for t in range(T):
        lpips_vals.append(_lpips_pair(lpips_model, ref[t], est[t]))

    return {
        "psnr_mean": float(np.nanmean(psnrs)),
        "ssim_mean": float(np.nanmean(ssims)),
        "lpips_mean": float(np.nanmean(lpips_vals)),
        "flicker_est": temporal_flicker(est),
        "frames_compared": float(T),
    }

def evaluate_video_only(est_dir: Path) -> Dict[str, float]:
    est = _to_float01(_read_frames(est_dir))
    return {
        "flicker_est": temporal_flicker(est),
        "num_frames": float(est.shape[0]),
    }


# -------------------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser(description="Video metrics (PSNR/SSIM/LPIPS + flicker).")
    ap.add_argument("--ref", type=Path, default=None, help="Reference frames dir")
    ap.add_argument("--est", type=Path, required=True, help="Estimated/generated frames dir")
    ap.add_argument("--lpips-device", type=str, default="cuda", help="Device for LPIPS (cuda/cpu)")
    args = ap.parse_args()

    if args.ref is not None:
        scores = evaluate_video_pair(args.ref, args.est, lpips_device=args.lpips_device)
    else:
        scores = evaluate_video_only(args.est)

    for k, v in scores.items():
        print(f"{k:14s}: {v:.4f}" if isinstance(v, float) else f"{k:14s}: {v}")

if __name__ == "__main__":
    main()
