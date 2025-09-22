#!/usr/bin/env python3
"""
extract_frames.py

Extract frames from videos, resize/center-crop to a target size, and (optionally)
materialize fixed-length clips by hardlinking/copying the selected frames.

It supports both:
  --input/--output              (primary flags)
  --input_path/--output_dir     (aliases, for convenience)

Examples
--------
# One-clip-per-video (good for GRID):
python scripts/extract_frames.py \
  --input data/video/GRID/raw/s1 \
  --output data/video/GRID/frames_tmp/s1 \
  --fps 16 --size 128 --clip-seconds 3 --hop-seconds 99 --keep-frames

# Recursive over a directory with mixed formats (.mpg/.mp4/.mov/...):
python scripts/extract_frames.py \
  --input data/video/GRID/raw \
  --output data/video/GRID/frames_tmp \
  --fps 16 --size 128
"""

from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm


# ----------------------------- formats & IO -----------------------------

class ImageFormat(str, Enum):
    JPG = ".jpg"
    PNG = ".png"
    WEBP = ".webp"

DEFAULT_IMG_FORMAT = ImageFormat.JPG

SUPPORTED_VIDEO_EXTS = (
    "mp4", "avi", "mov", "mkv", "mpg", "mpeg",
    "MP4", "AVI", "MOV", "MKV", "MPG", "MPEG"
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink if possible; otherwise copy."""
    ensure_dir(dst.parent)
    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ----------------------------- geometry -----------------------------

def parse_hw(size: str) -> Tuple[int, int]:
    """Parse '128' -> (128,128) or 'HxW' -> (H,W)."""
    s = size.lower()
    if "x" in s:
        h, w = s.split("x")
        return int(h), int(w)
    v = int(s)
    return v, v


def center_resize_crop(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Keep aspect, resize, then center-crop to (out_h, out_w)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid input frame with zero dimension.")

    scale = max(out_h / h, out_w / w)  # scale up so we can crop to target
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    y0 = max(0, (nh - out_h) // 2)
    x0 = max(0, (nw - out_w) // 2)
    cropped = resized[y0:y0 + out_h, x0:x0 + out_w]
    if cropped.shape[:2] != (out_h, out_w):
        cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)

    return cropped


# ----------------------------- indexing -----------------------------

def sample_indices(n_src_frames: int, src_fps: float, tgt_fps: float) -> List[int]:
    """Return frame indices to approximate target FPS from source FPS."""
    if n_src_frames <= 0:
        return []
    if tgt_fps <= 0 or tgt_fps >= src_fps:
        return list(range(n_src_frames))
    step = src_fps / tgt_fps
    idxs = [int(round(i * step)) for i in range(int(math.floor((n_src_frames - 1) / step)) + 1)]
    return [i for i in idxs if i < n_src_frames]


def chunk_ranges(n: int, clip_len: int, hop: int) -> List[Tuple[int, int]]:
    """Chunk [0,n) into windows of length clip_len with hop."""
    if n <= 0 or clip_len <= 0 or hop <= 0:
        return []
    if n < clip_len:
        return [(0, n)]
    out = []
    i = 0
    while i + clip_len <= n:
        out.append((i, i + clip_len))
        i += hop
    if not out:
        out = [(0, n)]
    return out


# ----------------------------- core logic -----------------------------

def gather_videos(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    vids: List[Path] = []
    for ext in SUPPORTED_VIDEO_EXTS:
        vids.extend(input_path.rglob(f"*.{ext}"))
    return vids


def extract_for_video(
    video_path: Path,
    out_root: Path,
    fps: float,
    size_hw: Tuple[int, int],
    clip_seconds: float,
    hop_seconds: float,
    keep_frames: bool,
    img_format: ImageFormat,
    quality: int,
) -> Dict:
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if src_fps <= 0:
            src_fps = 25.0  # sensible default for GRID if missing
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n_frames <= 0:
            raise RuntimeError(f"Could not read frame count for {video_path}")

        tgt_fps = fps if fps > 0 else src_fps
        keep = set(sample_indices(n_frames, src_fps, tgt_fps))

        name = video_path.stem
        out_dir = out_root / name
        frames_dir = out_dir / "frames"
        clips_dir = out_dir / "clips"
        ensure_dir(out_dir)
        if keep_frames:
            ensure_dir(frames_dir)
        ensure_dir(clips_dir)

        # Write params
        if img_format == ImageFormat.JPG:
            imwrite_flag = cv2.IMWRITE_JPEG_QUALITY
        elif img_format == ImageFormat.WEBP:
            imwrite_flag = cv2.IMWRITE_WEBP_QUALITY
        else:
            imwrite_flag = None  # PNG handled separately

        saved_paths: List[Path] = []
        frame_idx = 0
        saved = 0

        with tqdm(total=n_frames, desc=f"[video] {name}", unit="frame") as pbar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx in keep:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    proc = center_resize_crop(rgb, size_hw[0], size_hw[1])
                    out_name = f"frame_{saved:06d}{img_format.value}"
                    path = (frames_dir / out_name) if keep_frames else (out_dir / f"tmp_{out_name}")
                    if img_format == ImageFormat.PNG:
                        ok2 = cv2.imwrite(
                            str(path),
                            cv2.cvtColor(proc, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_PNG_COMPRESSION, 9],
                        )
                    else:
                        ok2 = cv2.imwrite(
                            str(path),
                            cv2.cvtColor(proc, cv2.COLOR_RGB2BGR),
                            [imwrite_flag, int(quality)],
                        )
                    if not ok2:
                        raise RuntimeError(f"Failed to write {path}")
                    saved_paths.append(path)
                    saved += 1
                frame_idx += 1
                pbar.update(1)

        if not saved_paths:
            return {
                "video": str(video_path),
                "out_dir": str(out_dir),
                "processed_frames": 0,
                "clips_created": 0,
                "target_fps": float(tgt_fps),
            }

        # Build clips by linking/copying frames
        clip_len = max(1, int(round(tgt_fps * clip_seconds)))
        hop = max(1, int(round(tgt_fps * hop_seconds)))
        ranges = chunk_ranges(len(saved_paths), clip_len, hop)

        manifest = {
            "video_name": name,
            "source_path": str(video_path),
            "resolution": [size_hw[0], size_hw[1]],
            "source_fps": float(src_fps),
            "target_fps": float(tgt_fps),
            "clip_seconds": float(clip_seconds),
            "hop_seconds": float(hop_seconds),
            "total_frames": n_frames,
            "processed_frames": len(saved_paths),
            "clips": [],
        }

        for ci, (a, b) in enumerate(ranges):
            cdir = clips_dir / f"clip_{ci:04d}"
            ensure_dir(cdir)
            for i in range(a, b):
                src = saved_paths[i]
                dst = cdir / f"frame_{i - a:06d}{img_format.value}"
                link_or_copy(src, dst)
            manifest["clips"].append({
                "clip_idx": ci,
                "start_frame": int(a),
                "end_frame": int(b),
                "num_frames": int(b - a),
                "start_sec": a / tgt_fps,
                "end_sec": b / tgt_fps,
                "frames_dir": str(cdir),
            })

        # Save manifest
        with open(out_dir / "clips.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Clean tmp frames if we didn’t keep
        if not keep_frames:
            for p in saved_paths:
                try:
                    p.unlink()
                except OSError:
                    pass

        return {
            "video": str(video_path),
            "out_dir": str(out_dir),
            "processed_frames": len(saved_paths),
            "clips_created": len(ranges),
            "target_fps": float(tgt_fps),
        }

    finally:
        cap.release()


# ----------------------------- CLI -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frames and build fixed-length clips.")
    # primary flags
    ap.add_argument("--input", type=Path, help="Input video file or directory")
    ap.add_argument("--output", type=Path, help="Output directory")
    # aliases (optional)
    ap.add_argument("--input_path", dest="input_alias", type=Path, help="Alias for --input")
    ap.add_argument("--output_dir", dest="output_alias", type=Path, help="Alias for --output")

    ap.add_argument("--fps", type=float, default=0.0, help="Target FPS (0 = use source FPS)")
    ap.add_argument("--size", type=str, default="128", help="Output size: '128' or 'HxW'")
    ap.add_argument("--clip-seconds", type=float, default=3.0, help="Clip duration (sec)")
    ap.add_argument("--hop-seconds", type=float, default=1.0, help="Hop between clips (sec)")
    ap.add_argument("--keep-frames", action="store_true", help="Keep all sampled frames in out_dir/frames")
    ap.add_argument("--format", type=str, choices=[f.value for f in ImageFormat], default=DEFAULT_IMG_FORMAT.value)
    ap.add_argument("--quality", type=int, default=95, help="JPEG/WEBP quality (1-100)")
    args = ap.parse_args()

    # normalize aliases
    if args.input is None and getattr(args, "input_alias", None) is not None:
        args.input = args.input_alias
    if args.output is None and getattr(args, "output_alias", None) is not None:
        args.output = args.output_alias

    if args.input is None or args.output is None:
        ap.error("Please supply --input/--output (or --input_path/--output_dir).")

    try:
        size_hw = parse_hw(args.size)
    except Exception as e:
        print(f"[fatal] bad --size: {e}", file=sys.stderr)
        sys.exit(2)

    vids = gather_videos(args.input)
    if not vids:
        print(f"[warn] no videos found under {args.input}", file=sys.stderr)
        sys.exit(0)

    img_fmt = ImageFormat(args.format)
    for v in vids:
        try:
            print(f"\nProcessing {v} ...")
            stats = extract_for_video(
                video_path=v,
                out_root=args.output,
                fps=float(args.fps),
                size_hw=size_hw,
                clip_seconds=float(args.clip_seconds),
                hop_seconds=float(args.hop_seconds),
                keep_frames=bool(args.keep_frames),
                img_format=img_fmt,
                quality=int(args.quality),
            )
            print(f" -> frames: {stats['processed_frames']}, clips: {stats['clips_created']}, out: {stats['out_dir']}")
        except Exception as e:
            print(f"[error] {v}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
