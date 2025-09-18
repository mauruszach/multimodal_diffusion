#!/usr/bin/env python3
"""
audio_codec.py — tiny audio encoder/decoder to/from latent frames.

Shapes
------
wav  : [B, 1, L]    (mono, range ~[-1, 1])
z_a  : [B, Ca, Fa]  (Ca = latent channels, Fa = latent frames)

Config (maps to your mvp.yaml)
------------------------------
audio:
  sr: 16000
  latent:
    channels: 8
    frame_hop_ms: 20        # preferred: hop in milliseconds (used to derive hop_samples)
    frames_per_clip: 150    # (optional) if set, encoder pools to EXACTLY this many frames
  codec:
    hop_samples: 320        # fallback if frame_hop_ms is absent (≈20 ms @ 16 kHz)
    hidden: 64
    smooth_kernel: 7
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- config -----------------------------

@dataclass
class AudioCodecConfig:
    in_ch: int = 1
    lat_ch: int = 8
    sr: int = 16000
    hop_samples: int = 320          # stride in samples between latent frames
    hidden: int = 64
    smooth_kernel: int = 7
    frames_per_clip: Optional[int] = None  # if provided, encoder produces exactly Fa frames

    @staticmethod
    def from_dict(d: Dict) -> "AudioCodecConfig":
        """
        Prefer `latent.frame_hop_ms`; else fallback to `codec.hop_samples` or 320.
        """
        lat = d.get("latent", {})
        codec = d.get("codec", {})
        sr = int(d.get("sr", 16000))

        # hop_samples derivation: prefer frame_hop_ms
        if "frame_hop_ms" in lat:
            hop_ms = float(lat["frame_hop_ms"])
            hop_samples = max(1, int(round(sr * hop_ms / 1000.0)))
        else:
            hop_samples = int(codec.get("hop_samples", 320))

        frames_per_clip = int(lat.get("frames_per_clip", 0)) or None

        return AudioCodecConfig(
            in_ch=int(d.get("in_ch", 1)),
            lat_ch=int(lat.get("channels", 8)),
            sr=sr,
            hop_samples=hop_samples,
            hidden=int(codec.get("hidden", 64)),
            smooth_kernel=int(codec.get("smooth_kernel", 7)),
            frames_per_clip=frames_per_clip,
        )


# --------------------------- building blocks ---------------------------

def _conv1d_block(c_in: int, c_out: int, k: int, act=nn.GELU, norm: bool = False) -> nn.Sequential:
    p = k // 2
    layers = [nn.Conv1d(c_in, c_out, kernel_size=k, padding=p), act()]
    if norm:
        layers.append(nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out))
    return nn.Sequential(*layers)


# ------------------------------ module --------------------------------

class AudioCodec(nn.Module):
    """
    Minimal learned codec that maps raw waveforms to a framewise latent and back.

    Encoder:
      wav [B,1,L] --pre conv--> [B,H,L] --avgpool stride=hop--> [B,H,Fa] --1x1--> z [B,Ca,Fa]
      If cfg.frames_per_clip is set, we pool to EXACTLY that many frames by adjusting
      the effective hop (with right-pad/crop) so that Fa matches.

    Decoder:
      z [B,Ca,Fa] --1x1--> [B,H,Fa] --nearest upsample by hop--> [B,H,L≈Fa*hop] --smooth convs--> wav̂ [B,1,L]
    """

    def __init__(self, cfg: AudioCodecConfig):
        super().__init__()
        self.cfg = cfg

        k = max(3, int(cfg.smooth_kernel))

        # Lightweight front-end
        self.pre = nn.Sequential(
            _conv1d_block(cfg.in_ch, cfg.hidden, k=9, act=nn.GELU, norm=False),
            _conv1d_block(cfg.hidden, cfg.hidden, k=9, act=nn.GELU, norm=False),
        )

        # Latent projections
        self.to_lat   = nn.Conv1d(cfg.hidden, cfg.lat_ch, kernel_size=1)
        self.from_lat = nn.Conv1d(cfg.lat_ch, cfg.hidden, kernel_size=1)

        # Synthesis smoothing
        pad = k // 2
        self.smooth = nn.Sequential(
            nn.Conv1d(cfg.hidden, cfg.hidden, kernel_size=k, padding=pad),
            nn.GELU(),
            nn.Conv1d(cfg.hidden, cfg.hidden, kernel_size=k, padding=pad),
            nn.GELU(),
            nn.Conv1d(cfg.hidden, cfg.in_ch, kernel_size=k, padding=pad),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------ helpers ------------------------------

    @classmethod
    def from_config(cls, d: Dict) -> "AudioCodec":
        return cls(AudioCodecConfig.from_dict(d))

    @property
    def hop(self) -> int:
        return int(self.cfg.hop_samples)

    @staticmethod
    def _compute_exact_pool_params(L: int, Fa: int) -> Tuple[int, int]:
        """
        Given an input length L and desired frames Fa, choose an integer hop so that
        total = Fa * hop >= L, and we can right-pad minimally.
        Returns (hop, total_len).
        """
        assert Fa > 0
        hop = max(1, int(round(L / Fa)))
        total = Fa * hop
        if total < L:
            hop += 1
            total = Fa * hop
        return hop, total

    def _avgpool_frames(self, x: torch.Tensor, target_Fa: Optional[int] = None) -> torch.Tensor:
        """
        Average-pool along time to produce [B,H,Fa].
        If target_Fa is None: Fa = ceil(L / hop); hop = cfg.hop_samples.
        If target_Fa is set:  pick hop so Fa matches exactly (with right padding).
        """
        B, H, L = x.shape
        if target_Fa is None:
            hop = self.hop
            Fa = math.ceil(L / hop)
            total = Fa * hop
        else:
            Fa = int(target_Fa)
            hop, total = self._compute_exact_pool_params(L, Fa)

        if total > L:
            x = F.pad(x, (0, total - L))
        elif total < L:
            x = x[..., :total]

        y = F.avg_pool1d(x, kernel_size=hop, stride=hop, ceil_mode=False, count_include_pad=False)  # [B,H,Fa]
        return y

    # ------------------------------- API ---------------------------------

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B,1,L] (mono)
        Returns z: [B, Ca, Fa]
        - If cfg.frames_per_clip is provided, Fa == frames_per_clip (exact).
        - Otherwise Fa ≈ ceil(L / hop_samples).
        """
        assert wav.dim() == 3 and wav.size(1) == 1, "AudioCodec.encode expects [B,1,L]"
        x = wav
        h = self.pre(x)  # [B,H,L]

        target_Fa = self.cfg.frames_per_clip
        h_f = self._avgpool_frames(h, target_Fa=target_Fa)  # [B,H,Fa]
        z = self.to_lat(h_f)  # [B,Ca,Fa]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B,Ca,Fa]  -> wav̂: [B,1,L≈Fa*hop]
        The output length is Fa * hop_samples (integer).
        """
        assert z.dim() == 3, "AudioCodec.decode expects [B,Ca,Fa]"
        B, Ca, Fa = z.shape

        h = self.from_lat(z)  # [B,H,Fa]
        # nearest upsample by fixed hop (keeps alignment with encoder’s default stride)
        L_out = Fa * self.hop
        h_up = F.interpolate(h, size=L_out, mode="nearest")  # [B,H,L]
        y = self.smooth(h_up)
        y = torch.tanh(y)  # keep in [-1,1]
        return y

    # --------------------------- diagnostics ---------------------------

    def check_consistency(self, clip_seconds: float | None = None):
        """
        Optional: sanity-check config against dataset timing.
        If frames_per_clip and clip_seconds are both provided, verify timing.
        """
        if self.cfg.frames_per_clip is None or clip_seconds is None:
            return

        hop_s = self.hop / float(self.cfg.sr)
        dur_est = self.cfg.frames_per_clip * hop_s
        want = float(clip_seconds)
        if abs(dur_est - want) > 0.02:  # >20 ms mismatch
            warnings.warn(
                f"[AudioCodec] frames_per_clip × hop_s = {dur_est:.3f}s "
                f"does not match clip_seconds={want:.3f}s. "
                f"Consider adjusting 'latent.frame_hop_ms' or 'latent.frames_per_clip'."
            )
