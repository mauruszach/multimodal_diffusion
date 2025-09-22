from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


# ----------------------------
# Beta schedules
# ----------------------------

def make_beta_schedule(
    steps: int,
    kind: str = "cosine",
    min_beta: float = 1e-4,
    max_beta: float = 2e-2,
) -> torch.Tensor:
    """
    Return betas[t] for t=0..steps-1.

    kind:
      - "cosine": Nichol & Dhariwal 2021 (improved DDPM)
      - "linear": linearly spaced betas between [min_beta, max_beta]
      - "sigmoid": sigmoid ramp between [min_beta, max_beta]
    """
    kind = kind.lower()
    if kind == "linear":
        betas = torch.linspace(min_beta, max_beta, steps, dtype=torch.float32)
        return betas.clamp(1e-8, 0.999)

    if kind == "sigmoid":
        xs = torch.linspace(-6, 6, steps, dtype=torch.float32)
        sig = torch.sigmoid(xs)
        betas = min_beta + (max_beta - min_beta) * sig
        return betas.clamp(1e-8, 0.999)

    if kind == "cosine":
        # Follow Nichol & Dhariwal (https://arxiv.org/abs/2102.09672)
        s = 0.008
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float32)  # 0..T
        f = torch.cos(((t / steps + s) / (1 + s)) * math.pi / 2) ** 2
        a_bar = f / f[0]  # normalize so that a_bar[0] = 1
        betas = 1 - (a_bar[1:] / a_bar[:-1])
        betas = betas.clamp(1e-8, 0.999)
        return betas

    raise ValueError(f"Unknown schedule kind: {kind}")


def alphas_cumprod_from_betas(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return alphas[t] and alpha_bar[t] (cumprod of alphas)."""
    betas = betas.to(dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar


# ----------------------------
# Timestep embedding (sinusoidal)
# ----------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
      timesteps: int tensor of shape [B]
      dim: embedding dimension (even number recommended)
      max_period: controls the minimum frequency of the embeddings

    Returns:
      Tensor of shape [B, dim]
    """
    if timesteps.dtype != torch.float32 and timesteps.dtype != torch.float64:
        timesteps = timesteps.float()

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        # pad one dimension if dim is odd
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


# ----------------------------
# Forward noising q(x_t | x_0)
# ----------------------------

def _gather(a_bar: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Gather alpha_bar[t] with broadcasting to match batch shape.
    t: (B,) int64/long
    returns: (B,) float32
    """
    if t.dtype != torch.long:
        t = t.long()
    return a_bar[t]


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    alpha_bar: torch.Tensor,
    eps: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample x_t = sqrt(ā_t) * x0 + sqrt(1 - ā_t) * epsilon.
    Returns (x_t, epsilon_used).
    """
    B = t.shape[0]
    a_bar_t = _gather(alpha_bar, t)                     # (B,)
    while a_bar_t.dim() < x0.dim():
        a_bar_t = a_bar_t.unsqueeze(-1)                 # (B,1,1,...)
    sqrt_ab = torch.sqrt(a_bar_t)
    sqrt_omb = torch.sqrt(torch.clamp(1.0 - a_bar_t, min=0.0))

    if eps is None:
        eps = torch.randn_like(x0)

    x_t = sqrt_ab * x0 + sqrt_omb * eps
    return x_t, eps


# ----------------------------
# DDIM sampling utilities
# ----------------------------

def make_sampling_schedule(T_train: int, T_sample: int) -> torch.Tensor:
    """
    Create a decreasing integer schedule of length T_sample+1 from (T_train-1) to -1.
    E.g., for T_train=1000, T_sample=10, you get 11 integers:
      [999, ~899, ..., ~0, -1]
    """
    grid = torch.linspace(T_train - 1, -1, T_sample + 1)
    # Round to nearest integer and cast to long
    sched = torch.round(grid).to(torch.long)
    # Ensure strictly non-increasing
    sched = torch.minimum(sched, torch.arange(T_sample, -1, -1, dtype=torch.long) * 0 + sched[0])
    return sched


def ddim_step(
    x_t: torch.Tensor,          # current sample at time t
    t_now: torch.Tensor,        # (B,) current integer timestep
    t_prev: torch.Tensor,       # (B,) previous integer timestep (can be -1)
    eps_hat: torch.Tensor,      # predicted noise ε̂(x_t, t)
    alpha_bar: torch.Tensor,    # ā[t] vector for all t
    eta: float = 0.0,
) -> torch.Tensor:
    """
    One DDIM update x_{t_prev} from x_t.

    Formulas (see DDIM paper; Nichol & Dhariwal re-derivation):
      x0_pred = (x_t - sqrt(1 - ā_t) * eps_hat) / sqrt(ā_t)
      sigma_t = eta * sqrt((1 - ā_{t-1})/(1 - ā_t) * (1 - ā_t/ā_{t-1}))
      x_{t-1} = sqrt(ā_{t-1}) * x0_pred
                + sqrt(1 - ā_{t-1} - sigma_t^2) * eps_hat
                + sigma_t * z,  z~N(0,I)
    We handle t_prev = -1 by setting ā_{-1} = 1.
    """
    # Gather ā_t and ā_{t-1}
    a_t = _gather(alpha_bar, torch.clamp(t_now, min=0))
    a_prev = torch.where(
        (t_prev >= 0), _gather(alpha_bar, torch.clamp(t_prev, min=0)), torch.ones_like(a_t)
    )  # ā_{-1} = 1

    # Broadcast to x_t shape
    def bcast(v: torch.Tensor) -> torch.Tensor:
        while v.dim() < x_t.dim():
            v = v.unsqueeze(-1)
        return v

    sqrt_a_t = torch.sqrt(bcast(a_t))
    sqrt_omb_t = torch.sqrt(torch.clamp(1.0 - bcast(a_t), min=0.0))
    sqrt_a_prev = torch.sqrt(bcast(a_prev))

    # Predict x0
    x0_pred = (x_t - sqrt_omb_t * eps_hat) / torch.clamp(sqrt_a_t, min=1e-8)

    # Sigma for DDIM (eta=0 => deterministic)
    if eta > 0.0:
        num = (1.0 - a_prev)
        den = (1.0 - a_t)
        frac = torch.clamp(num / torch.clamp(den, min=1e-8), min=0.0)
        one_minus_ratio = torch.clamp(1.0 - (a_t / torch.clamp(a_prev, min=1e-8)), min=0.0)
        sigma = eta * torch.sqrt(bcast(frac * one_minus_ratio))
    else:
        sigma = torch.zeros_like(x_t)

    # Second term coefficient
    coeff_eps = torch.sqrt(torch.clamp(1.0 - bcast(a_prev) - sigma ** 2, min=0.0))

    z = torch.randn_like(x_t) if eta > 0.0 else torch.zeros_like(x_t)

    x_prev = sqrt_a_prev * x0_pred + coeff_eps * eps_hat + sigma * z
    return x_prev
