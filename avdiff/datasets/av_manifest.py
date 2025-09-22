#!/usr/bin/env python3
# avdiff/datasets/av_manifest.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import soundfile as sf
import librosa
from PIL import Image

import torch
from torch.utils.data import Dataset


@dataclass
class AVManifestConfig:
    clip_seconds: float = 3.0
    fps: int = 16
    sr: int = 16000
    size_hw: Tuple[int, int] = (128, 128)  # (H, W)
    channels: int = 3                      # RGB


class AVManifestDataset(Dataset):
    """
    Dataset that reads a manifest (JSON) produced by tools/build_grid_manifest.py

    Each item in the manifest has:
      {
        "video_frames_dir": "path/to/clip_0000",
        "audio_wav_path":   "path/to/utterance.wav",
        "fps": 16,
        "sr": 16000,
        "clip_seconds": 3.0
      }

    __getitem__ returns:
      {
        "video": FloatTensor [3, T, H, W]  (in [0,1])
        "audio": FloatTensor [1, L]        (mono, in [-1,1])
        "fps":   int,
        "sr":    int,
        "video_frames_dir": str,
        "audio_wav_path":   str,
      }
    """
    def __init__(
        self,
        manifest_path: str | Path,
        clip_seconds: float = 3.0,
        fps: int = 16,
        sr: int = 16000,
        size_hw: Tuple[int, int] = (128, 128),
        channels: int = 3,
    ):
        self.manifest_path = Path(manifest_path)
        self.cfg = AVManifestConfig(
            clip_seconds=clip_seconds,
            fps=int(fps),
            sr=int(sr),
            size_hw=tuple(size_hw),
            channels=int(channels),
        )
        with open(self.manifest_path, "r") as f:
            m = json.load(f)
        # expect {"clips": [ ... ]}
        self.items: List[Dict] = m["clips"]

        self.T = int(round(self.cfg.fps * self.cfg.clip_seconds))
        self.L = int(round(self.cfg.sr * self.cfg.clip_seconds))

    def __len__(self) -> int:
        return len(self.items)

    # ---------- helpers ----------
    @staticmethod
    def _sorted_frames(dir_path: Path) -> List[Path]:
        # Expect names like frame_000000.jpg; sort lexicographically
        frames = sorted(dir_path.glob("frame_*.*"))
        if not frames:
            raise FileNotFoundError(f"No frames found under {dir_path}")
        return frames

    def _load_frames(self, frames_dir: Path) -> torch.Tensor:
        """Load up to T frames, resize if needed, return [3, T, H, W] in [0,1]."""
        H, W = self.cfg.size_hw
        frames = self._sorted_frames(frames_dir)

        need = self.T
        pick = frames[:min(len(frames), need)]
        if len(pick) < need:
            pick = pick + [frames[-1]] * (need - len(pick))

        imgs: List[np.ndarray] = []
        for p in pick:
            im = Image.open(p).convert("RGB")
            if im.size != (W, H):
                im = im.resize((W, H), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)  # [H, W, 3]
            imgs.append(arr)

        arr = np.stack(imgs, axis=0)               # [T, H, W, 3]
        arr = arr.astype(np.float32) / 255.0
        arr = np.transpose(arr, (3, 0, 1, 2))      # [3, T, H, W]
        return torch.from_numpy(arr)

    def _load_wav(self, wav_path: Path) -> torch.Tensor:
        """Load WAV, resample to sr, crop/pad to L samples, mono -> [1, L]."""
        y, orig_sr = sf.read(str(wav_path), always_2d=False)  # [L] or [L, C]
        if y.ndim == 2:
            y = y.mean(axis=1)  # to mono
        y = y.astype(np.float32, copy=False)

        if int(orig_sr) != self.cfg.sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.cfg.sr)

        L = self.L
        if y.shape[0] < L:
            pad = np.zeros((L - y.shape[0],), dtype=np.float32)
            y = np.concatenate([y, pad], axis=0)
        elif y.shape[0] > L:
            y = y[:L]

        y = y.reshape(1, -1)  # [1, L]
        return torch.from_numpy(y)

    # ---------- main ----------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | str]:
        item = self.items[idx]
        vdir = Path(item["video_frames_dir"])
        apath = Path(item["audio_wav_path"])

        vid = self._load_frames(vdir)   # [3, T, H, W]
        aud = self._load_wav(apath)     # [1, L]

        return {
            "video": vid,
            "audio": aud,
            "fps": int(self.cfg.fps),
            "sr": int(self.cfg.sr),
            "video_frames_dir": str(vdir),
            "audio_wav_path": str(apath),
        }


class AVClipsDataset(AVManifestDataset):
    """
    Thin wrapper to be compatible with training code that passes video_root/audio_root.
    If the manifest already contains full paths that exist, we use them as-is.
    Otherwise, if a root is provided, we try prefixing it.
    Extra kwargs are accepted and ignored safely.
    """
    def __init__(
        self,
        manifest_path: str | Path,
        clip_seconds: float = 3.0,
        fps: int = 16,
        sr: int = 16000,
        size_hw: Tuple[int, int] = (128, 128),
        channels: int = 3,
        video_root: Optional[str | Path] = None,
        audio_root: Optional[str | Path] = None,
        **_ignored,  # swallow any other unexpected kwargs
    ):
        super().__init__(
            manifest_path=manifest_path,
            clip_seconds=clip_seconds,
            fps=fps,
            sr=sr,
            size_hw=size_hw,
            channels=channels,
        )
        self.video_root = Path(video_root) if video_root else None
        self.audio_root = Path(audio_root) if audio_root else None

        # Normalize/resolve paths once at init.
        for it in self.items:
            vdir = Path(it["video_frames_dir"])
            apath = Path(it["audio_wav_path"])

            # Only change if current path doesn't exist.
            if not vdir.exists() and self.video_root is not None:
                cand = self.video_root / vdir
                if cand.exists():
                    it["video_frames_dir"] = str(cand)

            if not apath.exists() and self.audio_root is not None:
                cand = self.audio_root / apath
                if cand.exists():
                    it["audio_wav_path"] = str(cand)
