#!/usr/bin/env python3
"""
preprocess_audio.py

Loads audio files, (optionally) RMS-normalizes, computes log-mel spectrograms,
and chunks them into clips. It can either:
  1) Use fixed-duration windows (clip_seconds / hop_seconds), or
  2) Align to a video's clips.json (exact start/end seconds per clip).

Outputs (per audio file):
  out_root/
    <basename>/
      clips/
        clip_0000.npz   # {'mel': [n_mels, T], 'sr': sr, ...}
        clip_0001.npz
      clips.json        # manifest with clip metadata (start/end, paths)

Dependencies:
  - numpy
  - librosa
  - soundfile
  - tqdm
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_audio(path: Path, sr: int, mono: bool = True) -> np.ndarray:
    # librosa.load returns float32 in [-1, 1]
    y, _ = librosa.load(path, sr=sr, mono=mono)
    return y.astype(np.float32)


def rms_norm(y: np.ndarray, target_rms: float = 0.1, eps: float = 1e-8) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y * y) + eps))
    gain = target_rms / max(rms, eps)
    y = y * gain
    return np.clip(y, -1.0, 1.0)


def compute_logmel(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int],
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
    center: bool,
    eps: float = 1e-6,
) -> np.ndarray:
    if win_length is None:
        win_length = n_fft
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=center,
        power=2.0,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax or sr / 2,
    )
    logS = np.log(S + eps).astype(np.float32)  # [n_mels, frames]
    return logS


def chunk_by_seconds(n_samples: int, sr: int, clip_s: float, hop_s: float) -> List[Tuple[int, int]]:
    clip_len = int(round(sr * clip_s))
    hop = int(round(sr * hop_s))
    if clip_len <= 0:
        return [(0, n_samples)]
    ranges: List[Tuple[int, int]] = []
    if n_samples < clip_len:
        ranges.append((0, n_samples))
        return ranges
    start = 0
    while start + clip_len <= n_samples:
        ranges.append((start, start + clip_len))
        start += hop if hop > 0 else clip_len
    return ranges


def read_clip_manifest(align_to: Optional[Path]) -> Optional[Dict]:
    if align_to is None:
        return None
    try:
        with open(align_to, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read align-to manifest: {align_to} ({e})")
        return None


def save_clip_npz(
    out_path: Path,
    mel: np.ndarray,
    meta: Dict,
    save_wav: Optional[np.ndarray] = None,
    wav_sr: Optional[int] = None,
):
    out = {
        "mel": mel,  # [n_mels, T]
        "sr": meta["sr"],
        "n_fft": meta["n_fft"],
        "hop_length": meta["hop_length"],
        "win_length": meta["win_length"],
        "n_mels": meta["n_mels"],
        "fmin": meta["fmin"],
        "fmax": meta["fmax"],
        "center": meta["center"],
        "start_sec": meta["start_sec"],
        "end_sec": meta["end_sec"],
        "start_sample": meta["start_sample"],
        "end_sample": meta["end_sample"],
    }
    if save_wav is not None and wav_sr is not None:
        out["wav"] = save_wav.astype(np.float32)
        out["wav_sr"] = int(wav_sr)
    np.savez_compressed(out_path, **out)


def main():
    ap = argparse.ArgumentParser(description="Preprocess audio: log-mel features and clip chunking.")
    ap.add_argument("--in", dest="in_dir", type=Path, required=True, help="Input dir with audio files")
    ap.add_argument("--out", dest="out_dir", type=Path, required=True, help="Output root directory")
    ap.add_argument("--glob", type=str, default="**/*", help="Glob for audio files (default: **/*)")

    # Audio IO / normalization
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    ap.add_argument("--norm-rms", action="store_true", help="Apply simple RMS normalization to waveform")
    ap.add_argument("--save-wav", action="store_true", help="Also save per-clip waveform into the .npz")

    # Windowing / alignment
    ap.add_argument("--clip-seconds", type=float, default=3.0, help="Clip duration (seconds)")
    ap.add_argument("--hop-seconds", type=float, default=1.0, help="Hop between clips (seconds)")
    ap.add_argument("--align-to", type=Path, default=None, help="Path to video clips.json to align windows")

    # Mel config
    ap.add_argument("--n-mels", type=int, default=64, help="Number of mel bins")
    ap.add_argument("--n-fft", type=int, default=1024, help="FFT size")
    ap.add_argument("--hop-length", type=int, default=256, help="STFT hop length (samples)")
    ap.add_argument("--win-length", type=int, default=None, help="STFT window length (samples); default=n_fft")
    ap.add_argument("--fmin", type=float, default=20.0, help="Mel fmin")
    ap.add_argument("--fmax", type=float, default=None, help="Mel fmax (default: sr/2)")
    ap.add_argument("--center", action="store_true", help="Center STFT frames (librosa center=True)")
    args = ap.parse_args()

    audio_files = [p for p in args.in_dir.glob(args.glob) if p.suffix.lower() in AUDIO_EXTS and p.is_file()]
    if not audio_files:
        print(f"[WARN] No audio files found under {args.in_dir} (glob: {args.glob})")
        return

    ensure_dir(args.out_dir)

    align_manifest = read_clip_manifest(args.align_to)
    use_alignment = align_manifest is not None
    if use_alignment:
        clip_windows = [(clip["start_sec"], clip["end_sec"]) for clip in align_manifest["clips"]]
        clip_seconds = float(align_manifest.get("clip_seconds", 3.0))
        hop_seconds = float(align_manifest.get("hop_seconds", 1.0))
    else:
        clip_windows = None
        clip_seconds = args.clip_seconds
        hop_seconds = args.hop_seconds

    for apath in audio_files:
        base = apath.stem
        out_dir = args.out_dir / base
        clips_dir = out_dir / "clips"
        ensure_dir(out_dir)
        ensure_dir(clips_dir)

        # Load & normalize
        y = load_audio(apath, sr=args.sr, mono=True)
        if args.norm_rms:
            y = rms_norm(y)

        # Decide ranges
        if use_alignment:
            ranges = [(int(round(s * args.sr)), int(round(e * args.sr))) for (s, e) in clip_windows]
        else:
            ranges = chunk_by_seconds(len(y), args.sr, clip_seconds, hop_seconds)

        manifest = {
            "audio_name": base,
            "source_path": str(apath),
            "sr": args.sr,
            "clip_seconds": clip_seconds,
            "hop_seconds": hop_seconds,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "win_length": args.win_length if args.win_length is not None else args.n_fft,
            "n_mels": args.n_mels,
            "fmin": args.fmin,
            "fmax": args.fmax if args.fmax is not None else args.sr / 2,
            "center": bool(args.center),
            "clips": [],
        }

        with tqdm(total=len(ranges), desc=f"[audio] {base}", unit="clip") as pbar:
            for ci, (a, b) in enumerate(ranges):
                a = max(0, a)
                b = min(len(y), b)
                if b <= a:
                    pbar.update(1)
                    continue

                y_clip = y[a:b].copy()
                mel = compute_logmel(
                    y=y_clip,
                    sr=args.sr,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    win_length=args.win_length,
                    n_mels=args.n_mels,
                    fmin=args.fmin,
                    fmax=args.fmax,
                    center=args.center,
                )  # [n_mels, frames]

                clip_path = clips_dir / f"clip_{ci:04d}.npz"
                meta = {
                    "sr": args.sr,
                    "n_fft": args.n_fft,
                    "hop_length": args.hop_length,
                    "win_length": args.win_length if args.win_length is not None else args.n_fft,
                    "n_mels": args.n_mels,
                    "fmin": args.fmin,
                    "fmax": args.fmax if args.fmax is not None else args.sr / 2,
                    "center": bool(args.center),
                    "start_sec": a / args.sr,
                    "end_sec": b / args.sr,
                    "start_sample": a,
                    "end_sample": b,
                }

                if args.save_wav:
                    save_clip_npz(clip_path, mel=mel, meta=meta, save_wav=y_clip, wav_sr=args.sr)
                else:
                    save_clip_npz(clip_path, mel=mel, meta=meta)

                manifest["clips"].append({
                    "clip_idx": ci,
                    "start_sec": meta["start_sec"],
                    "end_sec": meta["end_sec"],
                    "num_samples": int(b - a),
                    "npz_path": str(clip_path),
                    "n_mels": int(args.n_mels),
                    "mel_frames": int(mel.shape[1]),
                })
                pbar.update(1)

        with open(out_dir / "clips.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[OK] wrote {out_dir / 'clips.json'}  | clips: {len(manifest['clips'])}")


if __name__ == "__main__":
    main()
