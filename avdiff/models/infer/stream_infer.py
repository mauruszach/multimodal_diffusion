#!/usr/bin/env python3
"""
stream_infer.py — sliding-window A→V or V→A generation with crossfade stitching.

It repeatedly calls the one-shot sampler on overlapping windows and
stitches outputs:
  - Audio: overlap-add crossfade
  - Video: alpha blend overlapping frames

Usage:
  # Long audio -> video
  python -m avdiff.infer.stream_infer \
      --config configs/mvp.yaml configs/a2v.yaml \
      --audio long.wav --out-dir out_vid --save-mp4 out.mp4

  # Long video -> audio
  python -m avdiff.infer.stream_infer \
      --config configs/mvp.yaml configs/v2a.yaml \
      --frames path/to/all_frames --out-wav out.wav
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch

from avdiff.utils.io import load_config, ensure_dir
from avdiff.utils import ops

# Reuse the sampler’s builders and sampling function
from avdiff.infer.sample_clip import (
    build_components, load_checkpoint_maybe, sample_one_direction,
    load_frames_dir, write_frames_and_optionally_mp4, load_audio_wav, save_audio_wav
)


def split_audio_into_windows(y: np.ndarray, sr: int, win_s: float, hop_s: float) -> Tuple[np.ndarray, int, int]:
    L = len(y)
    win = int(round(sr * win_s))
    hop = int(round(sr * hop_s))
    if L <= win:
        return y[None, :], win, hop
    chunks = []
    start = 0
    while start < L:
        end = min(L, start + win)
        seg = y[start:end]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        chunks.append(seg)
        if end == L:
            break
        start += hop
    return np.stack(chunks, axis=0), win, hop


def split_frames_into_windows(frames: np.ndarray, fps: int, win_s: float, hop_s: float) -> Tuple[np.ndarray, int, int]:
    T = frames.shape[0]
    win = int(round(fps * win_s))
    hop = int(round(fps * hop_s))
    if T <= win:
        return frames[None, ...], win, hop
    chunks = []
    start = 0
    while start < T:
        end = min(T, start + win)
        seg = frames[start:end]
        if seg.shape[0] < win:
            # pad by repeating last frame
            pad = np.repeat(seg[-1:], win - seg.shape[0], axis=0)
            seg = np.concatenate([seg, pad], axis=0)
        chunks.append(seg)
        if end == T:
            break
        start += hop
    return np.stack(chunks, axis=0), win, hop


def crossfade_audio(chunks: np.ndarray, sr: int, hop: int, win: int, fade_s: float) -> np.ndarray:
    """
    chunks: [N, L]
    returns stitched 1D array.
    """
    N, L = chunks.shape
    fade = int(round(sr * fade_s))
    if fade <= 0:
        # OLA with rectangular window
        y = np.zeros(( (N - 1) * hop + L,), dtype=np.float32)
        norm = np.zeros_like(y)
        for i in range(N):
            a = i * hop
            y[a:a+L] += chunks[i]
            norm[a:a+L] += 1.0
        norm = np.maximum(norm, 1e-6)
        return (y / norm).astype(np.float32)

    # Create cosine crossfade window
    w = np.ones(L, dtype=np.float32)
    w[:fade] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade, dtype=np.float32)))  # fade-in
    w[-fade:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade, dtype=np.float32))) # fade-out

    y = np.zeros(( (N - 1) * hop + L,), dtype=np.float32)
    norm = np.zeros_like(y)
    for i in range(N):
        a = i * hop
        y[a:a+L] += chunks[i] * w
        norm[a:a+L] += w
    norm = np.maximum(norm, 1e-6)
    return (y / norm).astype(np.float32)


def crossfade_video(chunks: np.ndarray, hop: int, win: int, fade_f: int) -> np.ndarray:
    """
    chunks: [N, T, H, W, 3] uint8
    returns stitched frames [T_total,H,W,3] uint8
    """
    N, L, H, W, C = chunks.shape
    out_T = (N - 1) * hop + L
    out = np.zeros((out_T, H, W, C), dtype=np.float32)
    norm = np.zeros((out_T, 1, 1, 1), dtype=np.float32)

    # Triangular fade ramp in frames
    fade = int(fade_f)
    if fade <= 0:
        w = np.ones((L, 1, 1, 1), dtype=np.float32)
    else:
        w = np.ones((L, 1, 1, 1), dtype=np.float32)
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        w[:fade] *= ramp.reshape(-1, 1, 1, 1)
        w[-fade:] *= ramp[::-1].reshape(-1, 1, 1, 1)

    for i in range(N):
        a = i * hop
        chunk = chunks[i].astype(np.float32) / 255.0
        out[a:a+L] += chunk * w
        norm[a:a+L] += w

    norm = np.maximum(norm, 1e-6)
    out = out / norm
    return (np.clip(out, 0, 1) * 255.0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Sliding-window AV generation with crossfade stitching.")
    ap.add_argument("--config", type=str, nargs="+", required=True, help="One or more YAML configs (merged).")
    ap.add_argument("--frames", type=Path, default=None, help="Prompt frames dir (for V→A).")
    ap.add_argument("--audio", type=Path, default=None, help="Prompt audio wav (for A→V).")
    ap.add_argument("--out-dir", type=Path, default=Path("stream_out"), help="Output directory (frames or .wav).")
    ap.add_argument("--save-mp4", type=Path, default=None, help="Optional mp4 path (for A→V).")
    ap.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = ap.parse_args()

    cfg = load_config(*args.config)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    prompt_modality = cfg.get("sampling", {}).get("prompt_modality", "video")
    win_s = float(cfg.get("streaming", {}).get("window_seconds", 3.0))
    hop_s = float(cfg.get("streaming", {}).get("hop_seconds", 1.0))
    xfade_s = float(cfg.get("streaming", {}).get("crossfade_seconds", 0.25))
    fps = int(cfg["video"]["fps"])
    sr = int(cfg["audio"]["sr"])

    # Build model components (shared across windows)
    vid_vae, aud_codec, adapt_v, adapt_a, core, head, tstep_dim = build_components(cfg, device)
    load_checkpoint_maybe(cfg, {
        "vid_vae": vid_vae, "aud_codec": aud_codec,
        "adapt_v": adapt_v, "adapt_a": adapt_a,
        "core": core, "head": head,
    })

    ensure_dir(args.out_dir)

    if prompt_modality == "video":
        if args.frames is None:
            raise SystemExit("Provide --frames for prompt_modality=video")
        frames_all = load_frames_dir(args.frames)  # [T,H,W,3] uint8
        chunks, win_f, hop_f = split_frames_into_windows(frames_all, fps=fps, win_s=win_s, hop_s=hop_s)
        out_audio_chunks = []

        for i in range(chunks.shape[0]):
            res = sample_one_direction(
                cfg=cfg, vid_vae=vid_vae, aud_codec=aud_codec, adapt_v=adapt_v, adapt_a=adapt_a,
                core=core, head=head, tstep_dim=tstep_dim,
                prompt_modality="video", prompt_video=chunks[i], prompt_audio=None, device=device
            )
            out_audio_chunks.append(res["audio"])

        # Stitch audio
        hop = int(round(sr * hop_s))
        win = int(round(sr * win_s))
        audio_mat = np.stack(out_audio_chunks, axis=0)  # [N, L]
        wav = crossfade_audio(audio_mat, sr=sr, hop=hop, win=win, fade_s=xfade_s)

        wav_path = args.out_dir / "stream_audio.wav"
        save_audio_wav(wav_path, wav, sr)
        print(f"[ok] wrote {wav_path}")

    else:
        if args.audio is None:
            raise SystemExit("Provide --audio for prompt_modality=audio")
        wav_all = load_audio_wav(args.audio, sr=sr)  # [L]
        chunks, win, hop = split_audio_into_windows(wav_all, sr=sr, win_s=win_s, hop_s=hop_s)
        out_video_chunks = []

        for i in range(chunks.shape[0]):
            res = sample_one_direction(
                cfg=cfg, vid_vae=vid_vae, aud_codec=aud_codec, adapt_v=adapt_v, adapt_a=adapt_a,
                core=core, head=head, tstep_dim=tstep_dim,
                prompt_modality="audio", prompt_video=None, prompt_audio=chunks[i], device=device
            )
            out_video_chunks.append(res["video"])

        # Stitch video chunks with crossfade (in frames)
        fade_f = int(round(xfade_s * fps))
        video_mat = np.stack(out_video_chunks, axis=0)  # [N, T, H, W, 3] uint8
        frames = crossfade_video(video_mat, hop=int(round(fps * hop_s)), win=int(round(fps * win_s)), fade_f=fade_f)

        frames_dir = args.out_dir / "frames"
        mp4 = args.save_mp4
        write_frames_and_optionally_mp4(frames, frames_dir, mp4_path=mp4, fps=fps)
        print(f"[ok] wrote frames → {frames_dir}")
        if mp4:
            print(f"[ok] wrote mp4 → {mp4}")


if __name__ == "__main__":
    main()
