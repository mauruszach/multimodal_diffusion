#!/usr/bin/env python3
"""
vae_video3d.py — simple 3D (time/space) autoencoder for video latents.

Design goals:
- Minimal dependencies (torch only)
- Deterministic by default (set variational=True to enable VAE reparameterization)
- Exact, configurable downsample factors:
    T' = T // t_down,  H' = H // s_down,  W' = W // s_down
- Safe handling of non-divisible inputs via center-crop (warns once)

API:
  VideoVAE.from_config(cfg_dict)
  encode(x: [B,3,T,H,W]) -> z: [B,Cv,T',H',W']
  decode(z) -> x_hat: [B,3,T,H,W]  (values in [0,1])

Config example (fits earlier scripts):
video:
  size: [256, 256]
  fps: 16
  latent:
    channels: 8
    t_down: 4
    s_down: 8
  encoder:
    base: 64
    blocks: 2
  decoder:
    base: 64
    blocks: 2
  variational: false
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


_warned_divisibility = False


@dataclass
class VideoVAEConfig:
    in_ch: int = 3
    lat_ch: int = 8
    t_down: int = 4
    s_down: int = 8
    enc_base: int = 64
    enc_blocks: int = 2
    dec_base: int = 64
    dec_blocks: int = 2
    variational: bool = False
    out_activation: str = "sigmoid"  # "sigmoid" or "tanh"

    @staticmethod
    def from_dict(d: Dict) -> "VideoVAEConfig":
        lat = d.get("latent", {})
        enc = d.get("encoder", {})
        dec = d.get("decoder", {})
        return VideoVAEConfig(
            in_ch=int(d.get("in_ch", 3)),
            lat_ch=int(lat.get("channels", 8)),
            t_down=int(lat.get("t_down", 4)),
            s_down=int(lat.get("s_down", 8)),
            enc_base=int(enc.get("base", 64)),
            enc_blocks=int(enc.get("blocks", 2)),
            dec_base=int(dec.get("base", 64)),
            dec_blocks=int(dec.get("blocks", 2)),
            variational=bool(d.get("variational", False)),
            out_activation=str(d.get("out_activation", "sigmoid")),
        )


def _conv_block_3d(c_in, c_out, ks=(3,3,3), act=nn.GELU, norm=True):
    pad = tuple(k // 2 for k in ks)
    layers = [nn.Conv3d(c_in, c_out, ks, padding=pad), act()]
    if norm:
        layers.append(nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out))
    return nn.Sequential(*layers)


class VideoVAE(nn.Module):
    def __init__(self, cfg: VideoVAEConfig):
        super().__init__()
        self.cfg = cfg

        # --- Encoder ---
        C = cfg.enc_base
        enc = [_conv_block_3d(cfg.in_ch, C)]
        for _ in range(cfg.enc_blocks - 1):
            enc.append(_conv_block_3d(C, C))
        # Downsample to latent grid exactly (avg pool)
        self.pool = nn.AvgPool3d(kernel_size=(cfg.t_down, cfg.s_down, cfg.s_down), stride=(cfg.t_down, cfg.s_down, cfg.s_down))
        self.enc_net = nn.Sequential(*enc)

        # Latent projection(s)
        if cfg.variational:
            self.to_mu   = nn.Conv3d(C, cfg.lat_ch, kernel_size=1)
            self.to_logv = nn.Conv3d(C, cfg.lat_ch, kernel_size=1)
        else:
            self.to_lat  = nn.Conv3d(C, cfg.lat_ch, kernel_size=1)

        # --- Decoder ---
        D = cfg.dec_base
        self.from_lat = nn.Conv3d(cfg.lat_ch, D, kernel_size=1)
        dec = []
        for _ in range(cfg.dec_blocks):
            dec.append(_conv_block_3d(D, D))
        self.dec_net = nn.Sequential(*dec)

        # Upsample back to input size exactly
        self.up_t = cfg.t_down
        self.up_s = cfg.s_down
        self.to_img = nn.Conv3d(D, cfg.in_ch, kernel_size=1)

        # Output activation
        if cfg.out_activation == "sigmoid":
            self.act_out = nn.Sigmoid()
        elif cfg.out_activation == "tanh":
            self.act_out = nn.Tanh()
        else:
            raise ValueError("out_activation must be 'sigmoid' or 'tanh'")

    # ---------- construction ----------

    @classmethod
    def from_config(cls, d: Dict) -> "VideoVAE":
        return cls(VideoVAEConfig.from_dict(d))

    # ---------- utilities ----------

    def _check_divisible(self, T: int, H: int, W: int) -> Tuple[int, int, int, Tuple[int,int,int,int,int,int]]:
        """
        Ensure T,H,W divisible by (t_down, s_down, s_down). If not, center-crop minimally.
        Returns (T2,H2,W2, crop_slices).
        """
        global _warned_divisibility
        t_down, s_down = self.cfg.t_down, self.cfg.s_down

        T2 = (T // t_down) * t_down
        H2 = (H // s_down) * s_down
        W2 = (W // s_down) * s_down

        if (T2, H2, W2) != (T, H, W):
            if not _warned_divisibility:
                warnings.warn(
                    f"[VideoVAE] Input (T={T},H={H},W={W}) not divisible by "
                    f"(t_down={t_down}, s_down={s_down}); center-cropping to (T={T2},H={H2},W={W2})."
                )
                _warned_divisibility = True
        # compute crop
        t0 = (T - T2) // 2
        h0 = (H - H2) // 2
        w0 = (W - W2) // 2
        return T2, H2, W2, (t0, t0 + T2, h0, h0 + H2, w0, w0 + W2)

    # ---------- forward API ----------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,T,H,W] in [0,1] or [-1,1]; returns z: [B,Cv,T',H',W'].
        """
        B, C, T, H, W = x.shape
        T2, H2, W2, (t0, t1, h0, h1, w0, w1) = self._check_divisible(T, H, W)
        if (T2, H2, W2) != (T, H, W):
            x = x[:, :, t0:t1, h0:h1, w0:w1]

        h = self.enc_net(x)                             # [B, Cenc, T,H,W]
        h_ds = self.pool(h)                             # [B, Cenc, T',H',W']
        if self.cfg.variational:
            mu = self.to_mu(h_ds)
            logv = self.to_logv(h_ds)
            if self.training:
                std = torch.exp(0.5 * logv)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu
            # cache KL terms for optional loss usage downstream
            self._kld = 0.5 * torch.mean(-1 - logv + mu.pow(2) + logv.exp())
        else:
            z = self.to_lat(h_ds)
            self._kld = None
        return z

    def kld_loss(self) -> Optional[torch.Tensor]:
        """Return last KL term if variational and encode() was called; else None."""
        return self._kld

    def decode(self, z: torch.Tensor, out_size: Optional[Tuple[int,int,int]] = None) -> torch.Tensor:
        """
        z: [B,Cv,T',H',W']  -> x_hat: [B,3,T,H,W] in [0,1] (sigmoid) or [-1,1] (tanh).
        If out_size is provided (T,H,W), it overrides the exact upsample target.
        """
        B, Cv, Tp, Hp, Wp = z.shape
        h = self.from_lat(z)                            # [B, D, Tp,Hp,Wp]
        # upsample
        if out_size is None:
            T = Tp * self.up_t
            H = Hp * self.up_s
            W = Wp * self.up_s
        else:
            T, H, W = out_size
        h_up = F.interpolate(h, size=(T, H, W), mode="trilinear", align_corners=False)
        h = self.dec_net(h_up)
        x = self.to_img(h)
        x = self.act_out(x)
        # ensure [0,1] if sigmoid or [-1,1] if tanh (already)
        return x
