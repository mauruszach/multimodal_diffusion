#!/usr/bin/env python3
"""
schedule_utils.py — diffusion schedules, q_sample, and DDIM stepping.
"""

from __future__ import annotations
from typing import Literal, Tuple

import math
import torch


ScheduleKind = Literal["linear", "cosine", "sigmoid"]


# ---------------------------
# Beta schedules
# ---------------------------

def make_beta_schedule(steps: int,
                       kind: ScheduleKind = "cosine",
                       min_beta: float = 1e-4,
                       max_beta: float = 2e-2,
                       s: float = 0.008) -> torch.Tensor:
    """
    Create a beta schedule (length = steps).

    - linear: linspace(min_beta, max_beta)
    - cosine: Nichol & Dhariwal (improved DDPM) via alpha_bar(t)
    - sigmoid: smooth start/end using a logistic in log-beta space
    """
    device = torch.device("cpu")

    if kind == "linear":
        betas = torch.linspace(min_beta, max_beta, steps, device=device)

    elif kind == "cosine":
        # alpha_bar(t) = cos^2( (t/T + s)/(1+s) * pi/2 )
        ts = torch.linspace(0, steps, steps + 1, device=device)
        f = (ts / steps + s) / (1 + s)
        alpha_bar = torch.cos(f * math.pi / 2) ** 2
        betas = torch.clamp(1 - (alpha_bar[1:] / alpha_bar[:-1]), min=min_beta, max=max_beta)

    elif kind == "sigmoid":
        # Interpolate betas in log-space with a sigmoid ramp
        t = torch.linspace(-6, 6, steps, device=device)
        sig = torch.sigmoid(t)
        betas = (min_beta + (max_beta - min_beta) * sig)

    else:
        raise ValueError(f"Unknown schedule kind: {kind}")

    return betas.float()


def alphas_cumprod_from_betas(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given betas, return (alphas, alphas_cumprod).
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod


# ---------------------------
# Index extract helper
# ---------------------------

def _extract_at(tensor_1d: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Gather values from a 1D tensor by timestep indices `t` and reshape to match `x`.
    """
    out = tensor_1d.gather(0, t.clamp(min=0, max=tensor_1d.numel() - 1))
    # reshape to broadcast: [B, 1, 1, ...]
    while out.dim() < x.dim():
        out = out.unsqueeze(-1)
    return out.to(x.dtype).to(x.device)


# ---------------------------
# Forward (q_sample)
# ---------------------------

def q_sample(z0: torch.Tensor,
             t: torch.Tensor,
             alphas_cumprod: torch.Tensor,
             noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample q(z_t | z0) = sqrt(a_bar_t) z0 + sqrt(1 - a_bar_t) eps
    Args:
      z0: clean latent [...]; dtype/device define outputs
      t:  [B] integer timesteps (0..T-1)
      alphas_cumprod: [T] cumulative alphas
      noise: optional noise tensor ~ N(0, I) (same shape as z0)
    Returns:
      (z_t, eps_used)
    """
    if noise is None:
        noise = torch.randn_like(z0)
    a_bar = _extract_at(alphas_cumprod, t, z0)
    z_t = torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * noise
    return z_t, noise


# ---------------------------
# DDIM stepping
# ---------------------------

def ddim_step(z_t: torch.Tensor,
              t: torch.Tensor,
              t_prev: torch.Tensor,
              eps_hat: torch.Tensor,
              alphas_cumprod: torch.Tensor,
              eta: float = 0.0) -> torch.Tensor:
    """
    One DDIM update from t -> t_prev (t_prev < t), vectorized over batch.

    eta=0 => deterministic DDIM
    eta>0 => adds noise according to variance schedule
    """
    a_t   = _extract_at(alphas_cumprod, t, z_t)         # a_bar_t
    a_tp  = _extract_at(alphas_cumprod, t_prev, z_t)    # a_bar_{t_prev}

    # Predict x0
    x0_hat = (z_t - torch.sqrt(1.0 - a_t) * eps_hat) / torch.sqrt(a_t).clamp(min=1e-12)

    # Direction to x_t
    sigma = eta * torch.sqrt((1 - a_tp) / (1 - a_t) * (1 - a_t / a_tp)).clamp(min=0.0)
    noise = torch.randn_like(z_t) if eta > 0 else torch.zeros_like(z_t)

    z_prev = torch.sqrt(a_tp) * x0_hat + torch.sqrt(1 - a_tp - sigma ** 2) * eps_hat + sigma * noise
    return z_prev


def make_sampling_schedule(T_train: int, T_sample: int) -> torch.Tensor:
    """
    Uniformly spaced timesteps from T_train-1 down to 0, length T_sample.
    Returns: [T_sample] int64
    """
    if T_sample <= 1:
        return torch.tensor([T_train - 1], dtype=torch.long)
    return torch.linspace(T_train - 1, 0, steps=T_sample, dtype=torch.long)


# ---------------------------
# Timestep embedding (sinusoidal)
# ---------------------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Sinusoidal embedding for scalar timesteps (like in diffusion/transformers).
    Args:
      t: [B] or any shape broadcastable to [B]
      dim: embedding dimension (even)
    Returns:
      emb: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t.float().unsqueeze(-1) * freqs  # [B, half]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:  # pad if odd
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb
#!/usr/bin/env python3
"""
schedule_utils.py — diffusion schedules, q_sample, and DDIM stepping.
"""

from __future__ import annotations
from typing import Literal, Tuple

import math
import torch


ScheduleKind = Literal["linear", "cosine", "sigmoid"]


# ---------------------------
# Beta schedules
# ---------------------------

def make_beta_schedule(steps: int,
                       kind: ScheduleKind = "cosine",
                       min_beta: float = 1e-4,
                       max_beta: float = 2e-2,
                       s: float = 0.008) -> torch.Tensor:
    """
    Create a beta schedule (length = steps).

    - linear: linspace(min_beta, max_beta)
    - cosine: Nichol & Dhariwal (improved DDPM) via alpha_bar(t)
    - sigmoid: smooth start/end using a logistic in log-beta space
    """
    device = torch.device("cpu")

    if kind == "linear":
        betas = torch.linspace(min_beta, max_beta, steps, device=device)

    elif kind == "cosine":
        # alpha_bar(t) = cos^2( (t/T + s)/(1+s) * pi/2 )
        ts = torch.linspace(0, steps, steps + 1, device=device)
        f = (ts / steps + s) / (1 + s)
        alpha_bar = torch.cos(f * math.pi / 2) ** 2
        betas = torch.clamp(1 - (alpha_bar[1:] / alpha_bar[:-1]), min=min_beta, max=max_beta)

    elif kind == "sigmoid":
        # Interpolate betas in log-space with a sigmoid ramp
        t = torch.linspace(-6, 6, steps, device=device)
        sig = torch.sigmoid(t)
        betas = (min_beta + (max_beta - min_beta) * sig)

    else:
        raise ValueError(f"Unknown schedule kind: {kind}")

    return betas.float()


def alphas_cumprod_from_betas(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given betas, return (alphas, alphas_cumprod).
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod


# ---------------------------
# Index extract helper
# ---------------------------

def _extract_at(tensor_1d: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Gather values from a 1D tensor by timestep indices `t` and reshape to match `x`.
    """
    out = tensor_1d.gather(0, t.clamp(min=0, max=tensor_1d.numel() - 1))
    # reshape to broadcast: [B, 1, 1, ...]
    while out.dim() < x.dim():
        out = out.unsqueeze(-1)
    return out.to(x.dtype).to(x.device)


# ---------------------------
# Forward (q_sample)
# ---------------------------

def q_sample(z0: torch.Tensor,
             t: torch.Tensor,
             alphas_cumprod: torch.Tensor,
             noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample q(z_t | z0) = sqrt(a_bar_t) z0 + sqrt(1 - a_bar_t) eps
    Args:
      z0: clean latent [...]; dtype/device define outputs
      t:  [B] integer timesteps (0..T-1)
      alphas_cumprod: [T] cumulative alphas
      noise: optional noise tensor ~ N(0, I) (same shape as z0)
    Returns:
      (z_t, eps_used)
    """
    if noise is None:
        noise = torch.randn_like(z0)
    a_bar = _extract_at(alphas_cumprod, t, z0)
    z_t = torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * noise
    return z_t, noise


# ---------------------------
# DDIM stepping
# ---------------------------

def ddim_step(z_t: torch.Tensor,
              t: torch.Tensor,
              t_prev: torch.Tensor,
              eps_hat: torch.Tensor,
              alphas_cumprod: torch.Tensor,
              eta: float = 0.0) -> torch.Tensor:
    """
    One DDIM update from t -> t_prev (t_prev < t), vectorized over batch.

    eta=0 => deterministic DDIM
    eta>0 => adds noise according to variance schedule
    """
    a_t   = _extract_at(alphas_cumprod, t, z_t)         # a_bar_t
    a_tp  = _extract_at(alphas_cumprod, t_prev, z_t)    # a_bar_{t_prev}

    # Predict x0
    x0_hat = (z_t - torch.sqrt(1.0 - a_t) * eps_hat) / torch.sqrt(a_t).clamp(min=1e-12)

    # Direction to x_t
    sigma = eta * torch.sqrt((1 - a_tp) / (1 - a_t) * (1 - a_t / a_tp)).clamp(min=0.0)
    noise = torch.randn_like(z_t) if eta > 0 else torch.zeros_like(z_t)

    z_prev = torch.sqrt(a_tp) * x0_hat + torch.sqrt(1 - a_tp - sigma ** 2) * eps_hat + sigma * noise
    return z_prev


def make_sampling_schedule(T_train: int, T_sample: int) -> torch.Tensor:
    """
    Uniformly spaced timesteps from T_train-1 down to 0, length T_sample.
    Returns: [T_sample] int64
    """
    if T_sample <= 1:
        return torch.tensor([T_train - 1], dtype=torch.long)
    return torch.linspace(T_train - 1, 0, steps=T_sample, dtype=torch.long)


# ---------------------------
# Timestep embedding (sinusoidal)
# ---------------------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Sinusoidal embedding for scalar timesteps (like in diffusion/transformers).
    Args:
      t: [B] or any shape broadcastable to [B]
      dim: embedding dimension (even)
    Returns:
      emb: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t.float().unsqueeze(-1) * freqs  # [B, half]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:  # pad if odd
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb
