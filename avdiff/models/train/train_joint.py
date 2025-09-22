#!/usr/bin/env python3
"""
train_joint.py — entrypoint to train the joint A↔V diffusion model.

Launch (single GPU):
  python -m avdiff.models.train.train_joint --config configs/mvp.yaml

Launch (DDP, 4 GPUs):
  torchrun --nproc_per_node=4 -m avdiff.models.train.train_joint --config configs/mvp.yaml
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from avdiff.utils.io import load_config
from .trainer import AVTrainer

# Dataset yields dicts with "video": [3,T,H,W] float in [0,1], "audio": [1,L] float.
from avdiff.datasets.av_manifest import AVClipsDataset


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank = 0
        world = 1
    return rank, world


def main():
    ap = argparse.ArgumentParser(description="Joint AV diffusion training.")
    ap.add_argument("--config", type=str, nargs="+", required=True, help="YAML config(s) merged left→right")
    args = ap.parse_args()

    cfg = load_config(*args.config)
    rank, world = setup_ddp()

    # Build datasets (manifest-based)
    train_ds = AVClipsDataset(
        manifest_path=cfg["data"]["train_split_glob"],
        video_root=cfg["paths"]["video_root"],
        audio_root=cfg["paths"]["audio_root"],
        fps=cfg["video"]["fps"],
        sr=cfg["audio"]["sr"],
        clip_seconds=cfg["data"]["clip_seconds"],
        size_hw=tuple(cfg["video"]["size"]),
    )
    # (optional) val_ds = AVClipsDataset(...)

    trainer = AVTrainer(cfg=cfg, dataset_train=train_ds, dataset_val=None, rank=rank, world_size=world)

    # Train loop
    max_steps = int(cfg["training"]["max_steps"])
    while trainer.state.step < max_steps:
        trainer.train_one()

    # Final checkpoint from rank 0
    if rank == 0:
        ckpt_path = Path(cfg["paths"]["ckpt_dir"]) / f"{cfg['experiment']}_final.pt"
        trainer.save_checkpoint(ckpt_path)
        print(f"[ok] saved final checkpoint → {ckpt_path}")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
