#!/usr/bin/env python3
"""
Minimal synthetic smoke training for the AV diffusion MVP.

This does NOT require your full repo; it stubs:
- latent encoders (we just start from random 'clean' latents)
- tokenization via utils.ops
- tiny Transformer core
- multimodal noise heads
- diffusion schedules and q_sample / DDIM updates

It runs a few training steps and prints the loss to prove the wiring works.

Run:
  python -m tests.smoke_train
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from avdiff.utils import ops
from avdiff.utils import schedule_utils as su
from avdiff.utils.io import load_config

# If your heads live elsewhere, adjust this import:
from avdiff.models.heads.noise_heads import MultiModalNoiseHead


# ---------- Tiny core (Transformer encoder) ----------

class TinyCore(nn.Module):
    def __init__(self, d_model=1024, n_layers=4, n_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # nn.Transformer expects [S, B, E]
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, d] -> [N, B, d] -> core -> [B, N, d]
        x = x.transpose(0, 1)
        y = self.enc(x)
        return y.transpose(0, 1)


# ---------- Simple adapters (per-modality projection to token width) ----------

class Adapter(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, d_in]
        return self.proj(tokens)


# ---------- Helper: build synthetic latents and tokens ----------

@dataclass
class Shapes:
    # video latent
    B: int = 2
    Cv: int = 8
    Tv: int = 12   # T' (e.g., 48 / 4)
    Hv: int = 16
    Wv: int = 16
    t_p: int = 2
    p: int = 4     # h = w = p

    # audio latent
    Ca: int = 8
    Fa: int = 150  # frames per clip (post-encoder)
    l_chunk: int = 4
    s_chunk: int = 4

    # token width
    d: int = 1024


def latents_to_tokens(z_v: torch.Tensor, z_a: torch.Tensor, shp: Shapes) -> Tuple[torch.Tensor, slice, slice, torch.Tensor, torch.Tensor]:
    """
    Convert latents to raw per-token vectors (before projection to d).
    Returns:
      X_raw:     [B, Ntot, d_raw?] (we'll project later)
      slice_vid: slice covering video tokens
      slice_aud: slice covering audio tokens
      tok_raw_vid: [B, Nv, C_v * t_p * p * p]
      tok_raw_aud: [B, Na, C_a * l_chunk]
    """
    B = z_v.size(0)

    # Video tokens: tube_patch -> [B, Nv, Cv * t_p * p * p]
    tok_raw_vid = ops.tube_patch_video(z_v, t=shp.t_p, h=shp.p, w=shp.p)  # [B, Nv, Dv_raw]
    Nv = tok_raw_vid.size(1)

    # Audio tokens: chunk_1d -> [B, Ca, Na, l_chunk] -> flatten C*len
    windows = ops.chunk_1d(z_a, length=shp.l_chunk, stride=shp.s_chunk, dim=-1)  # [B, Ca, Na, l]
    Na = windows.size(2)
    tok_raw_aud = windows.permute(0, 2, 1, 3).contiguous().view(B, Na, shp.Ca * shp.l_chunk)

    # Concat (raw)
    X_raw = torch.cat([tok_raw_vid, tok_raw_aud], dim=1)  # [B, Nv+Na, ...]
    slice_vid = slice(0, Nv)
    slice_aud = slice(Nv, Nv + Na)
    return X_raw, slice_vid, slice_aud, tok_raw_vid, tok_raw_aud


# ---------- Main smoke training ----------

def main():
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to read configs/mvp.yaml; fall back to defaults if missing
    try:
        cfg = load_config("configs/mvp.yaml")
        d_model = int(cfg.get("tokenizer", {}).get("width", 1024))
        # token dims (must match your config math)
        out_dim_vid = int(cfg.get("model", {}).get("heads", {}).get("video", {}).get("out_dim", 256))
        out_dim_aud = int(cfg.get("model", {}).get("heads", {}).get("audio", {}).get("out_dim", 32))
        sampler_steps = int(cfg.get("diffusion", {}).get("video", {}).get("sampler_steps", 50))
    except Exception:
        d_model = 1024
        out_dim_vid = 256
        out_dim_aud = 32
        sampler_steps = 50

    # Shapes (match earlier recommendations)
    shp = Shapes(d=d_model)

    # Build schedules
    T = 1000
    betas_v = su.make_beta_schedule(T, kind="cosine")
    betas_a = su.make_beta_schedule(T, kind="cosine")
    _, a_bar_v = su.alphas_cumprod_from_betas(betas_v)
    _, a_bar_a = su.alphas_cumprod_from_betas(betas_a)

    # Model parts
    adapter_vid = Adapter(d_in=shp.Cv * shp.t_p * shp.p * shp.p, d_out=shp.d).to(device)
    adapter_aud = Adapter(d_in=shp.Ca * shp.l_chunk, d_out=shp.d).to(device)
    core = TinyCore(d_model=shp.d, n_layers=4, n_heads=8, mlp_ratio=4.0, dropout=0.1).to(device)
    head = MultiModalNoiseHead(
        input_dims={"video": shp.d, "audio": shp.d},
        output_dims={"video": out_dim_vid, "audio": out_dim_aud},
        hidden_dim=512,
        num_shared_layers=2,
        num_modality_specific_layers=1,
        dropout=0.1,
        activation="gelu",
    ).to(device)

    params = list(adapter_vid.parameters()) + list(adapter_aud.parameters()) + list(core.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05)

    # A few synthetic steps
    steps = 6
    B = shp.B

    for step in range(1, steps + 1):
        # --- Create synthetic clean latents z0 (as if produced by encoders) ---
        z_v0 = torch.randn(B, shp.Cv, shp.Tv, shp.Hv, shp.Wv, device=device)
        z_a0 = torch.randn(B, shp.Ca, shp.Fa, device=device)

        # --- Sample per-modality timesteps and noise; build noisy latents ---
        t_v = torch.randint(0, T, (B,), device=device)
        t_a = torch.randint(0, T, (B,), device=device)

        z_vt, eps_v = su.q_sample(z_v0, t_v, a_bar_v)  # each: shape like z_v0
        z_at, eps_a = su.q_sample(z_a0, t_a, a_bar_a)

        # --- Tokenize both latents and then project to common width d ---
        X_raw, sl_vid, sl_aud, tok_raw_vid, tok_raw_aud = latents_to_tokens(z_vt, z_at, shp)
        X_vid = adapter_vid(tok_raw_vid)  # [B, Nv, d]
        X_aud = adapter_aud(tok_raw_aud)  # [B, Na, d]
        X = torch.cat([X_vid, X_aud], dim=1)

        # (Optional) add simple timestep embeddings per modality (broadcasted)
        # Here we skip for simplicity in the smoke test.

        # --- Core ---
        H = core(X)  # [B, Ntot, d]

        # --- Heads (predict noise for BOTH; we'll choose loss depending on target) ---
        H_vid, H_aud = H[:, sl_vid, :], H[:, sl_aud, :]
        eps_hat = head({"video": H_vid, "audio": H_aud})  # dict of [B, Nv, 256] / [B, Na, 32]

        # --- Build true per-token noise targets by tokenizing eps_v, eps_a similarly ---
        # Video eps tokens:
        eps_v_tok_raw = ops.tube_patch_video(eps_v, t=shp.t_p, h=shp.p, w=shp.p)  # [B, Nv, 256]
        # Audio eps tokens:
        eps_a_windows = ops.chunk_1d(eps_a, length=shp.l_chunk, stride=shp.s_chunk, dim=-1)  # [B, Ca, Na, l]
        eps_a_tok_raw = eps_a_windows.permute(0, 2, 1, 3).contiguous().view(B, H_aud.size(1), shp.Ca * shp.l_chunk)  # [B, Na, 32]

        # --- Choose target direction: 50/50 video or audio ---
        target_video = (step % 2 == 0)

        if target_video:
            loss = F.mse_loss(eps_hat["video"], eps_v_tok_raw)
            direction = "A→V (target=video)"
        else:
            loss = F.mse_loss(eps_hat["audio"], eps_a_tok_raw)
            direction = "V→A (target=audio)"

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        print(f"[step {step:02d}] {direction}  loss={loss.item():.5f}")

    print("✅ Smoke train finished OK.")


if __name__ == "__main__":
    main()
