#!/usr/bin/env python3
"""
io.py — safe load/save helpers for JSON, YAML, NumPy, and PyTorch.
"""

from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Union, Optional

try:
    import yaml
except Exception:
    yaml = None  # YAML is optional

import numpy as np
import torch


PathLike = Union[str, os.PathLike]


# ---------------------------
# Basic path utilities
# ---------------------------

def ensure_dir(path: PathLike) -> Path:
    """Create directory if missing; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: PathLike, text: str, encoding: str = "utf-8") -> None:
    """Write text atomically via a temp file + rename."""
    dest = Path(path)
    ensure_dir(dest.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dest.parent, encoding=encoding) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, dest)


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Write bytes atomically via a temp file + rename."""
    dest = Path(path)
    ensure_dir(dest.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=dest.parent) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, dest)


# ---------------------------
# JSON
# ---------------------------

def load_json(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: PathLike, obj: Any, *, indent: int = 2, sort_keys: bool = False) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=sort_keys))


# ---------------------------
# YAML (optional)
# ---------------------------

def load_yaml(path: PathLike) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml`")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: PathLike, obj: Dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml`")
    text = yaml.safe_dump(obj, sort_keys=False)
    atomic_write_text(path, text)


# ---------------------------
# NumPy / Torch
# ---------------------------

def save_npz(path: PathLike, **arrays: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    np.savez_compressed(path, **arrays)


def load_npz(path: PathLike) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def save_torch(path: PathLike, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    torch.save(obj, path)


def load_torch(path: PathLike, map_location: Optional[str] = None) -> Any:
    return torch.load(path, map_location=map_location or "cpu")


# ---------------------------
# Config helpers
# ---------------------------

def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dict `upd` into `base` (mutates and returns base).
    """
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(*paths: PathLike) -> Dict[str, Any]:
    """
    Load and deep-merge multiple YAML/JSON configs (left→right precedence).
    """
    cfg: Dict[str, Any] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() in {".yaml", ".yml"}:
            part = load_yaml(p)
        elif p.suffix.lower() == ".json":
            part = load_json(p)
        else:
            raise ValueError(f"Unsupported config format: {p}")
        deep_update(cfg, part or {})
    return cfg
