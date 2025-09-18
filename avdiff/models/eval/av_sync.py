#!/usr/bin/env python3
"""
av_sync.py — simple A/V sync proxy metrics.

Idea:
  - Compute a VIDEO motion envelope (per-frame "activity") from frame differences
    or optical flow (if OpenCV's Farneback is available).
  - Compute an AUDIO loudness/RMS envelope sampled at the video frame rate.
  - Cross-correlate the two envelopes to estimate the best lag and correlation.

Outputs:
  - lag_seconds: signed offset that best aligns audio to video (audio leads if lag<0).
  - max_corr:    correlation coefficient at that lag ([-1, 1]).

CLI:
  python -m avdiff.eval.av_sync --frames path/to/frames --audio path.wav --sr 16000 --fps 16
  # or with a .mp4 directly:
  python -m avdiff.eval.av_sync --video path/to/video.mp4 --audio path.wav --sr 16000
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import librosa
except Exception:
    librosa = None


# ---------------- I/O helpers ----------------

def _list_frames(frames_dir: Path, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    files = []
    for e in exts:
        files.extend(sorted(frames_dir.glob(f"*{e}")))
    return files

def _load_video_frames(frames_dir: Optional[Path] = None,
                       video_path: Optional[Path] = None) -> Tuple[np.ndarray, float]:
    """
    Returns (frames_uint8[h,w,3] list stacked -> [T,H,W,3], fps).
    If frames_dir is provided, fps must be provided by caller (set to >0).
    If video_path is provided, tries to read fps from file (requires OpenCV).
    """
    if frames_dir is not None:
        fps = None  # caller must pass --fps
        paths = _list_frames(frames_dir)
        if not paths:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
        frames = []
        for p in paths:
            img = cv2.imread(str(p)) if cv2 else None
            if img is None and cv2 is None:
                raise RuntimeError("OpenCV is required to read images.")
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        arr = np.stack(frames, axis=0)  # [T,H,W,3]
        return arr, fps

    if video_path is not None:
        if cv2 is None:
            raise RuntimeError("OpenCV required to read a video file.")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        arr = np.stack(frames, axis=0)  # [T,H,W,3]
        return arr, float(fps)

    raise ValueError("Either frames_dir or video_path must be provided.")


def _load_audio(audio_path: Path, sr: int) -> np.ndarray:
    if librosa is None:
        raise RuntimeError("librosa is required to load audio.")
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    return y.astype(np.float32)


# ------------- Envelope extraction -------------

def video_motion_envelope(frames: np.ndarray,
                          method: str = "diff",
                          flow_mag_clip: Optional[float] = None) -> np.ndarray:
    """
    Compute per-frame motion "energy".
      method="diff": sum |frame[t] - frame[t-1]| over pixels (fast, robust)
      method="flow": mean optical flow magnitude (requires OpenCV)
    Returns array of length T (one value per frame), with env[0]=env[1] (copy).
    """
    T = frames.shape[0]
    if T < 2:
        return np.zeros((T,), dtype=np.float32)
    gray = np.mean(frames.astype(np.float32), axis=3)  # [T,H,W]

    if method == "diff":
        diffs = np.abs(gray[1:] - gray[:-1])  # [T-1,H,W]
        env = diffs.reshape(diffs.shape[0], -1).mean(axis=1)
    elif method == "flow":
        if cv2 is None:
            raise RuntimeError("Optical flow requires OpenCV.")
        env_vals = []
        for t in range(1, T):
            prev = gray[t - 1].astype(np.uint8)
            nxt = gray[t].astype(np.uint8)
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            if flow_mag_clip:
                mag = np.clip(mag, 0, flow_mag_clip)
            env_vals.append(mag.mean())
        env = np.array(env_vals, dtype=np.float32)
    else:
        raise ValueError("Unknown method for video_motion_envelope")

    # pad first frame
    env = np.concatenate([env[:1], env], axis=0)
    # z-score
    m, s = float(env.mean()), float(env.std() + 1e-8)
    return ((env - m) / s).astype(np.float32)


def audio_rms_envelope(wav: np.ndarray, sr: int, fps: float) -> np.ndarray:
    """
    Compute audio RMS per video frame window: window=1/fps seconds, hop=window.
    Returns length ~ T_frames (floor).
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    win = max(1, int(round(sr / fps)))
    hop = win
    n = 1 + (len(wav) - win) // hop if len(wav) >= win else 1
    env = []
    for i in range(n):
        a = i * hop
        b = min(len(wav), a + win)
        seg = wav[a:b]
        rms = np.sqrt((seg ** 2).mean() + 1e-10)
        env.append(rms)
    env = np.array(env, dtype=np.float32)
    # z-score
    m, s = float(env.mean()), float(env.std() + 1e-8)
    return ((env - m) / s).astype(np.float32)


# ------------- Cross-correlation & lag -------------

def best_lag_and_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[int, float]:
    """
    Return (lag, corr) where 'lag' is y shifted relative to x (positive = y delayed).
    Uses normalized cross-correlation restricted to [-max_lag, max_lag].
    """
    L = min(len(x), len(y))
    x = x[:L] - x[:L].mean()
    y = y[:L] - y[:L].mean()
    x_std = x.std() + 1e-8
    y_std = y.std() + 1e-8
    best_corr = -1.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x_seg = x[-lag:L]
            y_seg = y[:L + lag]
        elif lag > 0:
            x_seg = x[:L - lag]
            y_seg = y[lag:L]
        else:
            x_seg = x
            y_seg = y
        if len(x_seg) < 3:
            continue
        corr = float(np.dot(x_seg, y_seg) / ((len(x_seg) - 1) * x_std * y_std))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag, best_corr


def estimate_av_sync(frames: np.ndarray,
                     wav: np.ndarray,
                     sr: int,
                     fps: float,
                     max_lag_seconds: float = 1.0,
                     method: str = "diff") -> Tuple[float, float]:
    """
    Compute lag (seconds) and correlation between video motion envelope and audio RMS envelope.
    Positive lag means AUDIO should be delayed to align with VIDEO.
    """
    v_env = video_motion_envelope(frames, method=method)
    a_env = audio_rms_envelope(wav, sr=sr, fps=fps)
    T = min(len(v_env), len(a_env))
    v_env, a_env = v_env[:T], a_env[:T]
    max_lag = int(round(max_lag_seconds * fps))
    lag_frames, corr = best_lag_and_corr(v_env, a_env, max_lag=max_lag)
    lag_seconds = lag_frames / float(fps)
    return lag_seconds, float(corr)


# -------------------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser(description="A/V sync proxy (motion vs. loudness envelope).")
    ap.add_argument("--frames", type=Path, default=None, help="Directory of frames (RGB images).")
    ap.add_argument("--video", type=Path, default=None, help="Video file (mp4, etc.).")
    ap.add_argument("--fps", type=float, default=0.0, help="FPS (required if using --frames).")
    ap.add_argument("--audio", type=Path, required=True, help="Audio file (wav etc.).")
    ap.add_argument("--sr", type=int, default=16000, help="Audio sample rate for loading.")
    ap.add_argument("--max-lag", type=float, default=1.0, help="Max absolute lag to search (seconds).")
    ap.add_argument("--method", type=str, default="diff", choices=["diff", "flow"], help="Video envelope method.")
    args = ap.parse_args()

    frames = None
    fps = args.fps
    if args.frames is not None:
        frames, _ = _load_video_frames(frames_dir=args.frames)
        if fps <= 0:
            raise SystemExit("Please provide --fps when using --frames.")
    elif args.video is not None:
        frames, fps = _load_video_frames(video_path=args.video)
    else:
        raise SystemExit("Provide either --frames or --video")

    wav = _load_audio(args.audio, sr=args.sr)
    lag_s, corr = estimate_av_sync(frames, wav, sr=args.sr, fps=fps, max_lag_seconds=args.max_lag, method=args.method)
    print(f"Estimated lag: {lag_s:+.3f} s  (audio should be delayed if positive)")
    print(f"Correlation  : {corr:.3f}")

if __name__ == "__main__":
    main()
