#!/usr/bin/env python3
"""
noise_heads.py — ε-prediction heads for diffusion.
Includes:
  - NoisePredictionHead: single-modality MLP ε-predictor
  - MultiModalNoiseHead: shared + modality-specific ε-predictors for video/audio
"""

from __future__ import annotations
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "NoisePredictionHead",
    "MultiModalNoiseHead",
]


# ----------------------------- helpers -----------------------------

def _make_activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unsupported activation: {name}")


def _init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ----------------------- single-modality head ----------------------

class NoisePredictionHead(nn.Module):
    """
    A straightforward MLP that maps hidden features → predicted noise ε.
    Works on any input shape that ends with feature width; flattens & reshapes.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        hidden_dim = int(hidden_dim or input_dim)
        act = _make_activation(activation)

        layers = []
        if num_layers <= 1:
            layers.append(nn.Linear(self.input_dim, self.output_dim))
        else:
            # input -> hidden
            layers += [nn.Linear(self.input_dim, hidden_dim), nn.LayerNorm(hidden_dim), act, nn.Dropout(dropout)]
            # hidden -> hidden (num_layers-2 times)
            for _ in range(max(0, num_layers - 2)):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), act, nn.Dropout(dropout)]
            # hidden -> output
            layers += [nn.Linear(hidden_dim, self.output_dim)]

        self.mlp = nn.Sequential(*layers)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., input_dim]
        return: [..., output_dim]
        """
        in_shape = x.shape
        x = x.reshape(-1, in_shape[-1])
        y = self.mlp(x)
        out_shape = list(in_shape[:-1]) + [self.output_dim]
        return y.reshape(*out_shape)


# ----------------------- multi-modality head -----------------------

class MultiModalNoiseHead(nn.Module):
    """
    Multi-modality ε-prediction with:
      - per-modality input projection to hidden_dim
      - optional SHARED trunk (same weights for all modalities)
      - modality-specific trunk + final linear to that modality's output_dim

    Args:
      input_dims:  {"video": d_model, "audio": d_model}
      output_dims: {"video": Dv, "audio": Da}  (per-token ε dims)
      hidden_dim:  trunk width
      num_shared_layers: blocks shared across modalities (Linear→LN→Act→Drop)
      num_modality_specific_layers: blocks per modality (same pattern); the
        LAST layer maps to output_dim for that modality.
      share_parameters: if True, share the *modality-specific hidden stack* across
        modalities (still uses per-modality final output Linear to match dims).
    """
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dims: Dict[str, int],
        hidden_dim: int = 512,
        num_shared_layers: int = 2,
        num_modality_specific_layers: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        share_parameters: bool = False,
    ):
        super().__init__()
        self.modalities = list(input_dims.keys())
        self.input_dims = {k: int(v) for k, v in input_dims.items()}
        self.output_dims = {k: int(v) for k, v in output_dims.items()}
        self.hidden_dim = int(hidden_dim)
        self.share_parameters = bool(share_parameters)
        act = _make_activation(activation)

        # Per-modality input projection
        self.input_proj = nn.ModuleDict({
            m: nn.Linear(self.input_dims[m], self.hidden_dim) for m in self.modalities
        })

        # Shared trunk (optional)
        def _shared_block():
            return nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                act,
                nn.Dropout(dropout),
            )

        if num_shared_layers <= 0:
            self.shared = None
        else:
            self.shared = nn.Sequential(*[_shared_block() for _ in range(num_shared_layers)])

        # Modality-specific trunks
        def _spec_block():
            return nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                act,
                nn.Dropout(dropout),
            )

        if num_modality_specific_layers <= 0:
            # direct hidden -> output
            self.spec = nn.ModuleDict({m: nn.Identity() for m in self.modalities})
        else:
            if self.share_parameters:
                self.shared_specific_trunk = nn.Sequential(*[_spec_block() for _ in range(num_modality_specific_layers - 1)]) \
                    if num_modality_specific_layers > 1 else nn.Identity()
                # Each modality still needs its own final output head
                self.spec = nn.ModuleDict({m: nn.Identity() for m in self.modalities})
            else:
                self.spec = nn.ModuleDict({
                    m: nn.Sequential(*[_spec_block() for _ in range(num_modality_specific_layers - 1)]) \
                        if num_modality_specific_layers > 1 else nn.Identity()
                    for m in self.modalities
                })

        # Final per-modality output (hidden_dim → output_dim[m])
        self.out_proj = nn.ModuleDict({
            m: nn.Linear(self.hidden_dim, self.output_dims[m]) for m in self.modalities
        })

        # Init
        self.apply(_init_linear)

    def get_output_dim(self, modality: str) -> int:
        return int(self.output_dims[modality])

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        inputs: {modality: [..., input_dims[modality]]}
        returns: {modality: [..., output_dims[modality]]} (or first value if return_dict=False)
        """
        outputs: Dict[str, torch.Tensor] = {}

        # Process each present modality path independently (tie via shared trunk weights)
        for m in self.modalities:
            if m not in inputs or inputs[m] is None:
                continue

            x = inputs[m]
            in_shape = x.shape
            x = x.reshape(-1, in_shape[-1])               # [*, d_in]
            x = self.input_proj[m](x)                     # [*, h]

            if self.shared is not None:
                x = self.shared(x)                        # [*, h]

            # Modality-specific trunk
            if self.share_parameters and hasattr(self, "shared_specific_trunk"):
                x = self.shared_specific_trunk(x)
            else:
                trunk = self.spec[m]
                if not isinstance(trunk, nn.Identity):
                    x = trunk(x)

            # Output projection
            x = self.out_proj[m](x)                       # [*, out_dim]
            out_shape = list(in_shape[:-1]) + [self.output_dims[m]]
            outputs[m] = x.view(*out_shape)

        if return_dict:
            return outputs
        # backward compatibility: return first tensor in modality order
        for m in self.modalities:
            if m in outputs:
                return outputs[m]
        # if nothing present, raise
        raise ValueError("No modalities found in inputs for MultiModalNoiseHead.forward")
