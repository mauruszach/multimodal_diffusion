#!/usr/bin/env python3
"""
trainer.py — joint A↔V training loop (DDP-friendly), with EMA and logging.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import os
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from avdiff.utils.io import ensure_dir, save_torch
from avdiff.utils import ops
from avdiff.utils import schedule_utils as su
from avdiff.train.losses import mse_targets_only, alignment_loss
from avdiff.train.collate import collate_batch
from avdiff.train.mask_schedule import Any2AnySchedule

# expected model modules
from avdiff.models.encoders.vae_video3d import VideoVAE
from avdiff.models.encoders.audio_codec import AudioCodec
from avdiff.models.mmdt import MMDiT
from avdiff.models.heads.noise_heads import MultiModalNoiseHead


# ---------- helpers ----------

class LinearAdapter(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = torch.nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def add_sinusoidal_timestep(tokens: torch.Tensor, t_scalar: torch.Tensor, dim: int) -> torch.Tensor:
    emb = su.timestep_embedding(t_scalar, dim=dim).to(tokens.device)
    return tokens + emb.unsqueeze(1)  # add (assumes tokens already width d and last dim matches)


class EMA:
    def __init__(self, module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: torch.nn.Module):
        for k, v in module.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, module: torch.nn.Module):
        module.load_state_dict(self.shadow, strict=False)


# ---------- trainer ----------

@dataclass
class TrainState:
    step: int = 0
    best_val: float = float("inf")


class AVTrainer:
    def __init__(
        self,
        cfg: Dict,
        dataset_train,
        dataset_val=None,
        rank: int = 0,
        world_size: int = 1,
        log_dir: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.rank = rank
        self.world = world_size
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu")
        self.mixed = cfg.get("mixed_precision", "fp32")
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.mixed in {"fp16", "bf16"}))

        # Dirs / logging
        out_root = Path(cfg["paths"]["out_root"])
        self.ckpt_dir = ensure_dir(cfg["paths"]["ckpt_dir"])
        self.samples_dir = ensure_dir(cfg["paths"]["samples_dir"])
        self.writer = None
        if rank == 0:
            self.writer = SummaryWriter(log_dir=cfg["paths"]["log_dir"])

        # Data
        collate_T = int(round(cfg["data"]["clip_seconds"] * cfg["video"]["fps"]))
        collate_L = int(round(cfg["data"]["clip_seconds"] * cfg["audio"]["sr"]))
        self.schedule = Any2AnySchedule(cfg["training"]["any2any_targets"])

        if world_size > 1:
            sampler = DistributedSampler(dataset_train, shuffle=True, drop_last=True)
        else:
            sampler = None

        self.loader = DataLoader(
            dataset_train,
            batch_size=int(cfg["data"]["batch_size"]),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(cfg["data"]["num_workers"]),
            pin_memory=bool(cfg["data"]["pin_memory"]),
            prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)),
            collate_fn=lambda items: collate_batch(items, T_target=collate_T, L_target=collate_L, pick_target=self.schedule.sample_target()),
            drop_last=True,
        )

        self.loader_val = None  # (left for the user to wire if desired)

        # Build modules
        self._build_modules()

        # Optimizer / sched
        opt_cfg = cfg["training"]["optimizer"]
        self.opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(opt_cfg["lr"]),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
            weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
            eps=float(opt_cfg.get("eps", 1e-8)),
        )

        self.grad_clip = float(cfg["training"].get("grad_clip_norm", 1.0))
        self.cfg_drop_prob = float(cfg["training"].get("cfg_drop_prob", 0.1))
        ema_cfg = cfg["training"].get("ema", {"use_ema": True, "decay": 0.999})
        self.use_ema = bool(ema_cfg.get("use_ema", True))
        self.ema = EMA(self.core, decay=float(ema_cfg.get("decay", 0.999))) if self.use_ema else None

        # Diffusion schedules
        self._build_schedules()

        # State
        self.state = TrainState()

    # ---- module builders ----

    def _build_modules(self):
        cfg = self.cfg
        d = int(cfg["tokenizer"]["width"])
        tstep_dim = int(cfg["embeddings"].get("timestep_dim", 256))
        self.tstep_dim = tstep_dim

        # Encoders/decoders
        self.vid_vae = VideoVAE.from_config(cfg["video"]).to(self.device)
        self.aud_codec = AudioCodec.from_config(cfg["audio"]).to(self.device)

        # Heads / core
        self.head = MultiModalNoiseHead(
            input_dims={"video": d, "audio": d},
            output_dims={
                "video": int(cfg["model"]["heads"]["video"]["out_dim"]),
                "audio": int(cfg["model"]["heads"]["audio"]["out_dim"]),
            },
            hidden_dim=int(cfg["model"]["heads"]["video"]["hidden_dim"]),
            num_shared_layers=2,
            num_modality_specific_layers=1,
            dropout=float(cfg["model"]["core"].get("dropout", 0.1)),
            activation=cfg["model"]["heads"]["video"].get("activation", "gelu"),
        ).to(self.device)

        self.core = MMDiT(**cfg["model"]["core"]).to(self.device)

        # Simple adapters (project raw token dims to d, then we ADD time embedding)
        self.adapt_v = LinearAdapter(
            d_in=int(cfg["model"]["heads"]["video"]["out_dim"]),
            d_out=d
        ).to(self.device)
        self.adapt_a = LinearAdapter(
            d_in=int(cfg["model"]["heads"]["audio"]["out_dim"]),
            d_out=d
        ).to(self.device)

        # DDP wrap core/head/adapters (encoders are eval-ish but we include them to allow finetuning if desired)
        if self.world > 1:
            self.core = DDP(self.core, device_ids=[self.rank], find_unused_parameters=False)
            self.head = DDP(self.head, device_ids=[self.rank], find_unused_parameters=False)
            self.adapt_v = DDP(self.adapt_v, device_ids=[self.rank], find_unused_parameters=False)
            self.adapt_a = DDP(self.adapt_a, device_ids=[self.rank], find_unused_parameters=False)
            self.vid_vae = DDP(self.vid_vae, device_ids=[self.rank], find_unused_parameters=True)
            self.aud_codec = DDP(self.aud_codec, device_ids=[self.rank], find_unused_parameters=True)

    def _build_schedules(self):
        cfg = self.cfg
        # Video
        self.T_v = int(cfg["diffusion"]["video"]["steps"])
        betas_v = su.make_beta_schedule(
            steps=self.T_v,
            kind=cfg["diffusion"]["video"]["schedule"],
            min_beta=cfg["diffusion"]["video"]["min_beta"],
            max_beta=cfg["diffusion"]["video"]["max_beta"],
        )
        _, self.a_bar_v = su.alphas_cumprod_from_betas(betas_v)

        # Audio
        self.T_a = int(cfg["diffusion"]["audio"]["steps"])
        betas_a = su.make_beta_schedule(
            steps=self.T_a,
            kind=cfg["diffusion"]["audio"]["schedule"],
            min_beta=cfg["diffusion"]["audio"]["min_beta"],
            max_beta=cfg["diffusion"]["audio"]["max_beta"],
        )
        _, self.a_bar_a = su.alphas_cumprod_from_betas(betas_a)

    # ---- train step ----

    def parameters(self):
        # encoders + core + adapters + head
        for m in [self.vid_vae, self.aud_codec, self.core, self.adapt_v, self.adapt_a, self.head]:
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    def _tokenize_video(self, z_v, t_p, p):
        return ops.tube_patch_video(z_v, t=t_p, h=p, w=p)  # [B, Nv, C*tp*p*p]

    def _tokenize_audio(self, z_a, l_chunk, s_chunk):
        windows = ops.chunk_1d(z_a, length=l_chunk, stride=s_chunk, dim=-1)  # [B, Ca, Na, l]
        B, Ca, Na, l = windows.shape
        return windows.permute(0, 2, 1, 3).contiguous().view(B, Na, Ca * l)

    def train_one(self) -> Tuple[float, float]:
        cfg = self.cfg
        t_p = int(cfg["tokenizer"]["video"]["tube"]["t"])
        p = int(cfg["tokenizer"]["video"]["tube"]["h"])
        l_chunk = int(cfg["tokenizer"]["audio"]["chunk"]["length"])
        s_chunk = int(cfg["tokenizer"]["audio"]["chunk"]["stride"])
        align_w = float(cfg["training"].get("align_loss_weight", 0.0))

        self.vid_vae.train()
        self.aud_codec.train()
        if isinstance(self.core, DDP):
            self.core.module.train()
        else:
            self.core.train()
        if isinstance(self.head, DDP):
            self.head.module.train()
        else:
            self.head.train()
        if isinstance(self.adapt_v, DDP):
            self.adapt_v.module.train()
            self.adapt_a.module.train()
        else:
            self.adapt_v.train()
            self.adapt_a.train()

        total_loss = 0.0
        total_align = 0.0

        for batch in self.loader:
            # Move to device
            video = batch["video"].to(self.device) if batch["video"] is not None else None  # [B,3,T,H,W]
            audio = batch["audio"].to(self.device) if batch["audio"] is not None else None  # [B,1,L]
            has_v = batch["has_video"].to(self.device)
            has_a = batch["has_audio"].to(self.device)
            target = batch["target"]

            B = has_v.size(0)

            # Encode to latents
            with torch.cuda.amp.autocast(enabled=(self.mixed in {"fp16", "bf16"})):
                z_v0 = self.vid_vae.encode(video) if video is not None else None  # [B,Cv',T',H',W']
                z_a0 = self.aud_codec.encode(audio) if audio is not None else None  # [B,Ca',Fa]

                # Sample timesteps
                t_v = torch.randint(0, self.T_v, (B,), device=self.device)
                t_a = torch.randint(0, self.T_a, (B,), device=self.device)

                # Add noise (q_sample)
                if z_v0 is not None:
                    z_vt, eps_v = su.q_sample(z_v0, t_v, self.a_bar_v)
                else:
                    z_vt = eps_v = None
                if z_a0 is not None:
                    z_at, eps_a = su.q_sample(z_a0, t_a, self.a_bar_a)
                else:
                    z_at = eps_a = None

                # Tokenize latents
                tok_v = self._tokenize_video(z_vt, t_p, p) if z_vt is not None else None
                tok_a = self._tokenize_audio(z_at, l_chunk, s_chunk) if z_at is not None else None

                # Tokenize true noise
                eps_tok_v = self._tokenize_video(eps_v, t_p, p) if eps_v is not None else None
                eps_tok_a = self._tokenize_audio(eps_a, l_chunk, s_chunk) if eps_a is not None else None

                # Adapters to width d
                Xv = self.adapt_v(tok_v) if tok_v is not None else None
                Xa = self.adapt_a(tok_a) if tok_a is not None else None

                # Add time embeddings (per-modality)
                if Xv is not None:
                    Xv = add_sinusoidal_timestep(Xv, t_v, self.tstep_dim)
                if Xa is not None:
                    Xa = add_sinusoidal_timestep(Xa, t_a, self.tstep_dim)

                # Concatenate tokens in fixed order [video ; audio]
                X_list = []
                if Xv is not None: X_list.append(Xv)
                if Xa is not None: X_list.append(Xa)
                X = torch.cat(X_list, dim=1)  # [B, Ntot, d]

                # Classifier-free token drop on CONDITIONS (drop the NON-target tokens)
                if self.cfg_drop_prob > 0.0:
                    if target == "video" and Xa is not None:
                        drop = (torch.rand(B, 1, 1, device=self.device) < self.cfg_drop_prob).float()
                        Xa = Xa * (1.0 - drop)
                        X = torch.cat([Xv, Xa], dim=1)
                    elif target == "audio" and Xv is not None:
                        drop = (torch.rand(B, 1, 1, device=self.device) < self.cfg_drop_prob).float()
                        Xv = Xv * (1.0 - drop)
                        X = torch.cat([Xv, Xa], dim=1)

                # Core
                H = self.core(X)  # [B, Ntot, d]

                # Slice back
                idx_v_end = Xv.size(1) if Xv is not None else 0
                Hv = H[:, :idx_v_end, :] if Xv is not None else None
                Ha = H[:, idx_v_end:, :] if Xa is not None else None

                # Heads -> eps_hat dict
                in_dict = {}
                if Hv is not None: in_dict["video"] = Hv
                if Ha is not None: in_dict["audio"] = Ha
                eps_hat = self.head(in_dict)

                # True eps dict for loss
                true_dict = {}
                if eps_tok_v is not None: true_dict["video"] = eps_tok_v
                if eps_tok_a is not None: true_dict["audio"] = eps_tok_a

                loss_main = mse_targets_only(eps_hat, true_dict, target=target)
                loss_align = alignment_loss(Hv, Ha, weight=align_w)

                loss = loss_main + loss_align

            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            if self.use_ema and isinstance(self.core, DDP):
                self.ema.update(self.core.module)
            elif self.use_ema:
                self.ema.update(self.core)

            total_loss += float(loss_main.detach().item())
            total_align += float(loss_align.detach().item())
            self.state.step += 1

            if self.rank == 0 and self.state.step % int(self.cfg["training"]["log_every"]) == 0:
                if self.writer:
                    self.writer.add_scalar("train/loss_main", total_loss / self.cfg["training"]["log_every"], self.state.step)
                    self.writer.add_scalar("train/loss_align", total_align / self.cfg["training"]["log_every"], self.state.step)
                total_loss = 0.0
                total_align = 0.0

            if self.state.step % int(self.cfg["training"]["ckpt_every"]) == 0 and self.rank == 0:
                self.save_checkpoint(self.ckpt_dir / f"step_{self.state.step}.pt")

            if self.state.step >= int(self.cfg["training"]["max_steps"]):
                break

        return total_loss, total_align

    # ---- checkpoint ----

    def save_checkpoint(self, path: Path):
        state = {
            "step": self.state.step,
            "core": (self.core.module if isinstance(self.core, DDP) else self.core).state_dict(),
            "head": (self.head.module if isinstance(self.head, DDP) else self.head).state_dict(),
            "adapt_v": (self.adapt_v.module if isinstance(self.adapt_v, DDP) else self.adapt_v).state_dict(),
            "adapt_a": (self.adapt_a.module if isinstance(self.adapt_a, DDP) else self.adapt_a).state_dict(),
            "vid_vae": (self.vid_vae.module if isinstance(self.vid_vae, DDP) else self.vid_vae).state_dict(),
            "aud_codec": (self.aud_codec.module if isinstance(self.aud_codec, DDP) else self.aud_codec).state_dict(),
            "opt": self.opt.state_dict(),
        }
        if self.use_ema and self.ema is not None:
            state["ema"] = self.ema.shadow
        ensure_dir(path.parent)
        save_torch(path, state)
        # also write/update "latest"
        save_torch(self.ckpt_dir / f"{self.cfg['experiment']}_latest.pt", state)
