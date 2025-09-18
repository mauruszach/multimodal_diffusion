#!/usr/bin/env python3
"""
Quick shape / smoke tests for utility ops, schedules, and heads.

Run:
  python -m tests.test_shapes
or with pytest:
  pytest -q tests/test_shapes.py
"""

import math
import unittest
import torch

from avdiff.utils import ops
from avdiff.utils import schedule_utils as su

# If your heads live elsewhere, adjust the import:
from avdiff.models.heads.noise_heads import NoisePredictionHead, MultiModalNoiseHead


class TestOpsAndSchedules(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_tube_patch_and_unpatch(self):
        B, C, T, H, W = 2, 8, 12, 16, 16
        t, h, w = 2, 4, 4
        z = torch.randn(B, C, T, H, W)

        tokens = ops.tube_patch_video(z, t, h, w)  # [B, N, C*t*h*w]
        N = (T // t) * (H // h) * (W // w)
        self.assertEqual(tokens.shape, (B, N, C * t * h * w))

        z2 = ops.tube_unpatch_video(tokens, C=C, T=T, H=H, W=W, t=t, h=h, w=w)
        self.assertTrue(torch.allclose(z, z2, atol=1e-6))

    def test_chunk_1d_and_overlap_add(self):
        B, C, L = 2, 8, 150
        length, stride = 4, 4
        x = torch.randn(B, C, L)

        windows = ops.chunk_1d(x, length=length, stride=stride)  # [B, C, N, length]
        N = (L - length) // stride + 1
        self.assertEqual(windows.shape, (B, C, N, length))

        # Reconstruct with perfect overlap (no windowing) should average identical segments
        y = ops.overlap_add_1d(windows, stride=stride, length=length, apply_hann=False)  # [B, C, L]
        self.assertEqual(y.shape, (B, C, (N - 1) * stride + length))

    def test_schedules_and_q_sample(self):
        T = 1000
        betas = su.make_beta_schedule(steps=T, kind="cosine")
        self.assertEqual(betas.shape, (T,))
        alphas, a_bar = su.alphas_cumprod_from_betas(betas)
        self.assertEqual(alphas.shape, (T,))
        self.assertEqual(a_bar.shape, (T,))

        B = 3
        z0 = torch.randn(B, 5, 7)  # arbitrary latent shape
        t = torch.randint(0, T, (B,))
        zt, eps = su.q_sample(z0, t, a_bar)
        self.assertEqual(zt.shape, z0.shape)
        self.assertEqual(eps.shape, z0.shape)

        # One DDIM step (vectorized)
        schedule = su.make_sampling_schedule(T_train=T, T_sample=10)
        t_now = schedule[0].expand(B)
        t_prev = schedule[1].expand(B)
        eps_hat = torch.randn_like(zt)
        z_prev = su.ddim_step(zt, t_now, t_prev, eps_hat, a_bar, eta=0.0)
        self.assertEqual(z_prev.shape, z0.shape)


class TestHeads(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def test_noise_prediction_head(self):
        B, N, d_in, d_out = 2, 13, 1024, 256
        x = torch.randn(B, N, d_in)
        head = NoisePredictionHead(input_dim=d_in, output_dim=d_out, hidden_dim=512, num_layers=3, activation="gelu")
        y = head(x)
        self.assertEqual(y.shape, (B, N, d_out))

    def test_multimodal_noise_head(self):
        B, Nv, Na = 2, 96, 37
        d = 1024
        out_v, out_a = 256, 32

        h_vid = torch.randn(B, Nv, d)
        h_aud = torch.randn(B, Na, d)

        head = MultiModalNoiseHead(
            input_dims={"video": d, "audio": d},
            output_dims={"video": out_v, "audio": out_a},
            hidden_dim=512,
            num_shared_layers=2,
            num_modality_specific_layers=1,
            dropout=0.1,
            activation="gelu",
        )
        out = head({"video": h_vid, "audio": h_aud})
        self.assertIn("video", out)
        self.assertIn("audio", out)
        self.assertEqual(out["video"].shape, (B, Nv, out_v))
        self.assertEqual(out["audio"].shape, (B, Na, out_a))


if __name__ == "__main__":
    unittest.main()
