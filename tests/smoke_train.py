#!/usr/bin/env python3
"""
Minimal synthetic smoke training for the AV diffusion MVP.

This stubs:
  • synthetic clean latents z0_v (video) and z0_a (audio)
  • forward noising q(x_t | x_0) with per-modality schedules
  • tokenization (tube patches for video; 1D chunks for audio)
  • per-modality linear adapters to shared width d
  • tiny Transformer encoder core (shared denoiser)
  • multimodal heads predicting per-token noise in RAW spaces
  • single-target MSE (alternate A→V / V→A) to exercise backprop

Run:
  python tests/smoke_train.py --config configs/mvp.yaml --steps 50
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from avdiff.utils import ops
from avdiff.utils import schedule_utils as su
from avdiff.utils.io import load_config
from avdiff.models.heads.noise_heads import MultiModalNoiseHead


# -------------------------
# Tiny Transformer core
# -------------------------

class TinyCore(nn.Module):
    def __init__(self, d_model=1024, n_layers=4, n_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=False,   # expects [S, B, E]
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, d] -> [N, B, d] -> core -> [B, N, d]
        x = x.transpose(0, 1)
        y = self.enc(x)
        return y.transpose(0, 1)


# -------------------------
# Simple adapters (per-mod)
# -------------------------

class Adapter(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, d_in] -> [B, N, d_out]
        return self.proj(tokens)


# -------------------------
# Synthetic shapes (MVP)
# -------------------------

@dataclass
class Shapes:
    # Video latent (matches VAE downsample in configs/mvp.yaml)
    B: int = 2
    Cv: int = 8
    Tv: int = 12   # 48 frames @16fps, t_down=4 → 12
    Hv: int = 16   # 128/8
    Wv: int = 16   # 128/8
    t_p: int = 2
    p: int = 4     # tube patch: (t=2, h=4, w=4) → per-token 8*2*4*4 = 256 dims

    # Audio latent (codec frames per ~3s clip)
    Ca: int = 8
    Fa: int = 150  # ~3 s with ~16 ms hop
    l_chunk: int = 4
    s_chunk: int = 4  # per-token 8*4 = 32 dims

    # Shared token width for the core
    d: int = 1024


# -------------------------
# Latents → tokens helper
# -------------------------

def latents_to_tokens(
    z_v: torch.Tensor, z_a: torch.Tensor, shp: Shapes
) -> Tuple[None, slice, slice, torch.Tensor, torch.Tensor]:
    """
    Convert latents to raw per-token vectors (before projection to width d).

    Returns:
      X_raw         -> None (we do NOT concat raw streams of different dims)
      slice_vid     slice covering video tokens (assuming concat order vid+aud)
      slice_aud     slice covering audio tokens
      tok_raw_vid   [B, Nv, 256]  with 256 = Cv * t_p * p * p
      tok_raw_aud   [B, Na, 32]   with  32 = Ca * l_chunk
    """
    B = z_v.size(0)

    # Video tokens: tube patches
    tok_raw_vid = ops.tube_patch_video(
        z_v, t=shp.t_p, h=shp.p, w=shp.p
    )  # [B, Nv, Cv * t_p * p * p] = [B, Nv, 256]
    Nv = tok_raw_vid.size(1)

    # Audio tokens: chunk along frames and flatten Ca*l_chunk
    windows = ops.chunk_1d(
        z_a, length=shp.l_chunk, stride=shp.s_chunk, dim=-1
    )  # [B, Ca, Na, l_chunk]
    Na = windows.size(2)
    tok_raw_aud = (
        windows.permute(0, 2, 1, 3)
        .contiguous()
        .view(B, Na, shp.Ca * shp.l_chunk)
    )  # [B, Na, 32]

    # No raw concat here (dims differ). We'll project each to d and concat later.
    return None, slice(0, Nv), slice(Nv, Nv + Na), tok_raw_vid, tok_raw_aud


# -------------------------
# Utilities
# -------------------------

def pick_device(cli_device: str = "auto") -> torch.device:
    if cli_device != "auto":
        return torch.device(cli_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------
# Main (training loop)
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/mvp.yaml")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    # Load config (tolerant if fields are missing)
    try:
        cfg = load_config(args.config)
    except Exception:
        cfg = {}

    d_model = int(cfg.get("tokenizer", {}).get("width", 1024))
    shp = Shapes(d=d_model)

    # Raw per-token dims (must match heads’ out_dim)
    Dv_raw = shp.Cv * shp.t_p * shp.p * shp.p  # 8*2*4*4 = 256
    Da_raw = shp.Ca * shp.l_chunk              # 8*4      = 32

    # Schedules (per modality)
    T_v = int(cfg.get("diffusion", {}).get("video", {}).get("steps", 1000))
    T_a = int(cfg.get("diffusion", {}).get("audio", {}).get("steps", 1000))
    kind_v = cfg.get("diffusion", {}).get("video", {}).get("schedule", "cosine")
    kind_a = cfg.get("diffusion", {}).get("audio", {}).get("schedule", "cosine")

    betas_v = su.make_beta_schedule(T_v, kind=kind_v)
    betas_a = su.make_beta_schedule(T_a, kind=kind_a)
    _, a_bar_v = su.alphas_cumprod_from_betas(betas_v)
    _, a_bar_a = su.alphas_cumprod_from_betas(betas_a)

    # Ensure schedules live on the active device (fixes MPS/CUDA indexing issue)
    a_bar_v = a_bar_v.to(device)
    a_bar_a = a_bar_a.to(device)

    # Modules
    adapter_vid = Adapter(d_in=Dv_raw, d_out=shp.d).to(device)
    adapter_aud = Adapter(d_in=Da_raw, d_out=shp.d).to(device)
    core = TinyCore(d_model=shp.d, n_layers=4, n_heads=8, mlp_ratio=4.0, dropout=0.1).to(device)

    # Heads predict noise in RAW token spaces (so loss shapes match exactly)
    head = MultiModalNoiseHead(
        input_dims={"video": shp.d, "audio": shp.d},
        output_dims={"video": Dv_raw, "audio": Da_raw},
        hidden_dim=512,
        num_shared_layers=2,
        num_modality_specific_layers=1,
        dropout=0.1,
        activation="gelu",
    ).to(device)

    params = list(adapter_vid.parameters()) + list(adapter_aud.parameters()) \
             + list(core.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05)

    B = shp.B
    steps = int(args.steps)

    print(f"[smoke] device={device.type} d={shp.d} Dv_raw={Dv_raw} Da_raw={Da_raw} steps={steps}")

    for step in range(1, steps + 1):
        # --- synthetic clean latents (as if encoders produced them) ---
        z_v0 = torch.randn(B, shp.Cv, shp.Tv, shp.Hv, shp.Wv, device=device)
        z_a0 = torch.randn(B, shp.Ca, shp.Fa, device=device)

        # --- per-modality timesteps and forward noising ---
        t_v = torch.randint(0, T_v, (B,), device=device)
        t_a = torch.randint(0, T_a, (B,), device=device)
        z_vt, eps_v = su.q_sample(z_v0, t_v, a_bar_v)  # shapes match z_v0
        z_at, eps_a = su.q_sample(z_a0, t_a, a_bar_a)  # shapes match z_a0

        # --- tokenize noisy latents (raw token features) ---
        _, sl_vid, sl_aud, tok_v_raw, tok_a_raw = latents_to_tokens(z_vt, z_at, shp)  # [B,Nv,256], [B,Na,32]

        # --- project to shared width and concat token streams ---
        X_v = adapter_vid(tok_v_raw)  # [B, Nv, d]
        X_a = adapter_aud(tok_a_raw)  # [B, Na, d]
        X = torch.cat([X_v, X_a], dim=1)  # [B, Nv+Na, d]

        # --- core transformer ---
        H = core(X)                     # [B, Nv+Na, d]
        H_vid, H_aud = H[:, sl_vid, :], H[:, sl_aud, :]

        # --- heads predict per-token noise in RAW spaces ---
        eps_hat = head({"video": H_vid, "audio": H_aud})  # dict: "video":[B,Nv,256], "audio":[B,Na,32]

        # --- build true per-token noise targets by tokenizing eps_v, eps_a ---
        eps_v_tok = ops.tube_patch_video(eps_v, t=shp.t_p, h=shp.p, w=shp.p)  # [B, Nv, 256]
        eps_a_win = ops.chunk_1d(eps_a, length=shp.l_chunk, stride=shp.s_chunk, dim=-1)  # [B,Ca,Na,l]
        Na = eps_a_win.size(2)
        eps_a_tok = eps_a_win.permute(0, 2, 1, 3).contiguous().view(B, Na, Da_raw)  # [B, Na, 32]

        # --- pick target (alternate deterministically) and compute loss only for that modality ---
        target_video = bool((step % 2) == 0)  # even steps: video, odd: audio
        if target_video:
            loss = F.mse_loss(eps_hat["video"], eps_v_tok)
            direction = "A→V (target=video)"
        else:
            loss = F.mse_loss(eps_hat["audio"], eps_a_tok)
            direction = "V→A (target=audio)"

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        print(f"[step {step:03d}] {direction}  loss={loss.item():.6f}")

    print("✅ Smoke train finished OK.")


if __name__ == "__main__":
    main()
