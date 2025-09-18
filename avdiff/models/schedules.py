#!/usr/bin/env python3
"""
schedules.py — thin per-modality diffusion schedule utilities.

This wraps the generic utilities in avdiff.utils.schedule_utils into a
light object with convenient methods and a simple builder from config.

Usage:
  sch_v = ModalitySchedule.make(kind="cosine", steps=1000, min_beta=1e-4, max_beta=2e-2)
  z_t, eps = sch_v.q_sample(z0, t)
  z_prev   = sch_v.ddim_step(z_t, t, t_prev, eps_hat, eta=0.0)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal

import torch

from avdiff.utils import schedule_utils as su


ScheduleKind = Literal["linear", "cosine", "sigmoid"]


@dataclass
class ModalitySchedule:
    """
    A convenience wrapper over betas/alphas/alpha_bar with forward & reverse ops.
    """
    kind: ScheduleKind
    steps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor

    @classmethod
    def make(cls,
             *,
             kind: ScheduleKind = "cosine",
             steps: int = 1000,
             min_beta: float = 1e-4,
             max_beta: float = 2e-2) -> "ModalitySchedule":
        betas = su.make_beta_schedule(steps=steps, kind=kind, min_beta=min_beta, max_beta=max_beta)
        alphas, alpha_bar = su.alphas_cumprod_from_betas(betas)
        return cls(kind=kind, steps=int(steps), betas=betas, alphas=alphas, alphas_cumprod=alpha_bar)

    def to(self, device: torch.device) -> "ModalitySchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    # ---------- forward process ----------

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """
        q(z_t | z0) = sqrt(ᾱ_t) z0 + sqrt(1-ᾱ_t) ε
        Returns (z_t, ε_used)
        """
        return su.q_sample(z0, t, self.alphas_cumprod, noise=noise)

    # ---------- reverse process (DDIM) ----------

    def ddim_step(self,
                  z_t: torch.Tensor,
                  t: torch.Tensor,
                  t_prev: torch.Tensor,
                  eps_hat: torch.Tensor,
                  eta: float = 0.0) -> torch.Tensor:
        """
        One DDIM update from t → t_prev using predicted ε.
        """
        return su.ddim_step(z_t, t, t_prev, eps_hat, self.alphas_cumprod, eta=eta)

    # ---------- schedules for sampling ----------

    def make_sampling_schedule(self, steps_sample: int) -> torch.Tensor:
        """
        Uniform integer schedule from (T_train-1) → 0 with length steps_sample.
        """
        return su.make_sampling_schedule(self.steps, steps_sample)

    # ---------- utilities ----------

    def timestep_embedding(self, t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
        return su.timestep_embedding(t, dim=dim, max_period=max_period)


def build_schedules_from_config(cfg: Dict) -> Dict[str, ModalitySchedule]:
    """
    Convenience: build {"video": ModalitySchedule, "audio": ModalitySchedule} from YAML config.
    """
    v_cfg = cfg["diffusion"]["video"]
    a_cfg = cfg["diffusion"]["audio"]

    sch_v = ModalitySchedule.make(
        kind=v_cfg.get("schedule", "cosine"),
        steps=int(v_cfg.get("steps", 1000)),
        min_beta=float(v_cfg.get("min_beta", 1e-4)),
        max_beta=float(v_cfg.get("max_beta", 2e-2)),
    )
    sch_a = ModalitySchedule.make(
        kind=a_cfg.get("schedule", "cosine"),
        steps=int(a_cfg.get("steps", 1000)),
        min_beta=float(a_cfg.get("min_beta", 1e-4)),
        max_beta=float(a_cfg.get("max_beta", 2e-2)),
    )
    return {"video": sch_v, "audio": sch_a}
