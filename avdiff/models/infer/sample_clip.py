#!/usr/bin/env python3
"""
sample_clip.py — one-shot DDIM sampling with classifier-free guidance (CFG)
in either direction: prompt=video ⇒ audio, or prompt=audio ⇒ video.

Usage examples:
  # Video -> Audio
  python -m avdiff.infer.sample_clip \
      --config configs/mvp.yaml configs/v2a.yaml \
      --frames path/to/frames_dir --out-audio out.wav

  # Audio -> Video
  python -m avdiff.infer.sample_clip \
      --config configs/mvp.yaml configs/a2v.yaml \
      --audio path/to/clip.wav --out-frames out_frames --save-mp4 out.mp4
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

# I/O & utils
from avdiff.utils.io import load_config, ensure_dir, load_torch
from avdiff.utils import ops
from avdiff.utils import schedule_utils as su

# ---- Expected model modules (implement these in your repo) ----
# Encoders / decoders (latents)
from avdiff.models.encoders.vae_video3d import VideoVAE          # E_vid / D_vid
from avdiff.models.encoders.audio_codec import AudioCodec        # E_aud / D_aud

# Core & heads
from avdiff.models.mmdt import MMDiT                             # shared Transformer denoiser
from avdiff.models.heads.noise_heads import MultiModalNoiseHead  # per-modality noise heads

# (If you already built dedicated adapter/position/timestep modules, import them here.)
# In this script we inline a simple adapter to keep the dependency surface small.


# -------------------- Simple adapters --------------------

class LinearAdapter(torch.nn.Module):
    """Per-modality linear projection to token width d."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = torch.nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, d_in]
        return self.proj(x)


def add_sinusoidal_timestep(tokens: torch.Tensor,
                            t_scalar: torch.Tensor,
                            dim: int) -> torch.Tensor:
    """
    Add (via concat) a broadcasted sinusoidal timestep embedding per token.
    tokens: [B, N, d]
    t_scalar: [B] integer timesteps
    returns: [B, N, d + dim]
    """
    emb = su.timestep_embedding(t_scalar, dim=dim).to(tokens.device)  # [B, dim]
    emb = emb.unsqueeze(1).expand(-1, tokens.size(1), -1)             # [B, N, dim]
    return torch.cat([tokens, emb], dim=-1)


# -------------------- Builders --------------------

def build_components(cfg: Dict, device: torch.device):
    """Instantiate encoders/decoders, adapters, core, heads from config."""
    # Shapes / token dims
    d = int(cfg["tokenizer"]["width"])
    # Video per-token out dim
    out_v = int(cfg["model"]["heads"]["video"]["out_dim"])
    out_a = int(cfg["model"]["heads"]["audio"]["out_dim"])

    # --- Encoders / decoders ---
    vid_vae = VideoVAE.from_config(cfg["video"]).to(device).eval()
    aud_codec = AudioCodec.from_config(cfg["audio"]).to(device).eval()

    # --- Token adapters (raw token -> width d [+ timestep emb if we concat]) ---
    # We’ll concat a timestep embedding of size cfg["embeddings"]["timestep_dim"].
    tstep_dim = int(cfg["embeddings"].get("timestep_dim", 256))
    # Video token raw dim = out_v (C' * t_p * h * w), audio token raw dim = out_a (C' * l_chunk)
    adapt_v = LinearAdapter(out_v, d - tstep_dim).to(device)
    adapt_a = LinearAdapter(out_a, d - tstep_dim).to(device)

    # --- Core denoiser (DiT) ---
    core_cfg = cfg["model"]["core"]
    core = MMDiT(**core_cfg).to(device).eval()

    # --- Heads ---
    head = MultiModalNoiseHead(
        input_dims={"video": d, "audio": d},
        output_dims={"video": out_v, "audio": out_a},
        hidden_dim=int(cfg["model"]["heads"]["video"]["hidden_dim"]),
        num_shared_layers=2,
        num_modality_specific_layers=1,
        dropout=float(cfg["model"]["core"].get("dropout", 0.1)),
        activation=cfg["model"]["heads"]["video"].get("activation", "gelu"),
    ).to(device).eval()

    return vid_vae, aud_codec, adapt_v, adapt_a, core, head, tstep_dim


def load_checkpoint_maybe(cfg: Dict, modules: Dict[str, torch.nn.Module]) -> None:
    ckpt_path = cfg.get("paths", {}).get("ckpt_path") or cfg.get("paths", {}).get("ckpt")
    if not ckpt_path:
        print("[info] no ckpt_path provided in config; sampling with random weights.")
        return
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = load_torch(ckpt_path)
    # Allow flexible keys: try strict load where possible
    for name, mod in modules.items():
        key = f"{name}_state_dict"
        if isinstance(state, dict) and key in state:
            missing, unexpected = mod.load_state_dict(state[key], strict=False)
            print(f"[ckpt] loaded {name} (missing={len(missing)} unexpected={len(unexpected)})")
    # If you saved a flat state_dict, you can also try:
    if isinstance(state, dict) and "state_dict" in state and hasattr(modules, "core"):
        try:
            modules["core"].load_state_dict(state["state_dict"], strict=False)
        except Exception:
            pass


# -------------------- I/O helpers --------------------

def load_frames_dir(frames_dir: Path) -> np.ndarray:
    import cv2
    paths = sorted([p for p in frames_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read {p}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)  # [T,H,W,3]


def write_frames_and_optionally_mp4(frames_uint8: np.ndarray, out_dir: Path, mp4_path: Optional[Path] = None, fps: int = 16):
    import cv2
    ensure_dir(out_dir)
    T, H, W, _ = frames_uint8.shape
    for t in range(T):
        cv2.imwrite(str(out_dir / f"frame_{t:06d}.jpg"), cv2.cvtColor(frames_uint8[t], cv2.COLOR_RGB2BGR))
    if mp4_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(mp4_path), fourcc, fps, (W, H))
        for t in range(T):
            vw.write(cv2.cvtColor(frames_uint8[t], cv2.COLOR_RGB2BGR))
        vw.release()


def load_audio_wav(path: Path, sr: int) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def save_audio_wav(path: Path, wav: np.ndarray, sr: int):
    import soundfile as sf
    ensure_dir(Path(path).parent)
    sf.write(str(path), wav, sr)


# -------------------- Tokenization --------------------

def latents_to_tokens_video(z_v: torch.Tensor, t_p: int, p: int) -> torch.Tensor:
    # z_v: [B,Cv',T',H',W'] -> [B, Nv, Cv'*t_p*p*p]
    return ops.tube_patch_video(z_v, t=t_p, h=p, w=p)


def latents_to_tokens_audio(z_a: torch.Tensor, l_chunk: int, s_chunk: int) -> torch.Tensor:
    # z_a: [B,Ca',Fa] -> windows [B, Ca', Na, l_chunk] -> [B, Na, Ca'*l_chunk]
    windows = ops.chunk_1d(z_a, length=l_chunk, stride=s_chunk, dim=-1)  # [B, Ca', Na, l]
    B, Ca, Na, l = windows.shape
    return windows.permute(0, 2, 1, 3).contiguous().view(B, Na, Ca * l)   # [B, Na, Ca*l]


def tokens_to_latents_audio(tokens: torch.Tensor, Ca: int, l_chunk: int, Fa: int, stride: int) -> torch.Tensor:
    """
    Inverse of latents_to_tokens_audio for the final x0 aggregation:
    We reconstruct per-channel sequences via overlap-add of chunk windows predicted as noise or x0 pieces.
    Here we use a simple folding that matches chunk/stride.
    """
    B, Na, D = tokens.shape
    assert D == Ca * l_chunk
    windows = tokens.view(B, Na, Ca, l_chunk).permute(0, 2, 1, 3).contiguous()  # [B, Ca, Na, l]
    # Overlap-add back to [B, Ca, L]
    recons = []
    for b in range(B):
        ch = []
        for c in range(Ca):
            y = ops.overlap_add_1d(windows[b, c].unsqueeze(0), stride=stride, length=l_chunk, apply_hann=False)
            ch.append(y)
        ch = torch.stack(ch, dim=1)  # [1, Ca, L]
        recons.append(ch)
    z = torch.cat(recons, dim=0)  # [B, Ca, L]
    # Clip to Fa length if needed
    if z.size(-1) > Fa:
        z = z[..., :Fa]
    elif z.size(-1) < Fa:
        z = F.pad(z, (0, Fa - z.size(-1)))
    return z


# -------------------- DDIM with CFG (one target) --------------------

@torch.no_grad()
def sample_one_direction(
    *,
    cfg: Dict,
    vid_vae: VideoVAE,
    aud_codec: AudioCodec,
    adapt_v: LinearAdapter,
    adapt_a: LinearAdapter,
    core: MMDiT,
    head: MultiModalNoiseHead,
    tstep_dim: int,
    prompt_modality: str,      # "video" or "audio"
    prompt_video: Optional[np.ndarray],   # [T,H,W,3] uint8 if video prompting
    prompt_audio: Optional[np.ndarray],   # [L] float32 if audio prompting
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Perform DDIM sampling to generate the other modality.
    Returns dict with either "audio" (wav float32) or "video" (frames uint8).
    """
    # ---- Config bits ----
    d = int(cfg["tokenizer"]["width"])
    Ttrain_v = int(cfg["diffusion"]["video"]["steps"])
    Ttrain_a = int(cfg["diffusion"]["audio"]["steps"])
    S_v = int(cfg["diffusion"]["video"]["sampler_steps"])
    S_a = int(cfg["diffusion"]["audio"]["sampler_steps"])
    eta = float(cfg["sampling"].get("ddim_eta", 0.0))
    g_scale_v = float(cfg["sampling"]["guidance_scale"].get("video", 3.0))
    g_scale_a = float(cfg["sampling"]["guidance_scale"].get("audio", 3.0))

    # Latent/tokens sizing
    t_p = int(cfg["tokenizer"]["video"]["tube"]["t"])
    p = int(cfg["tokenizer"]["video"]["tube"]["h"])
    l_chunk = int(cfg["tokenizer"]["audio"]["chunk"]["length"])
    s_chunk = int(cfg["tokenizer"]["audio"]["chunk"]["stride"])

    # Video latent shape
    Cv = int(cfg["video"]["latent"]["channels"])
    t_down = int(cfg["video"]["latent"]["t_down"])
    s_down = int(cfg["video"]["latent"]["s_down"])
    # Audio latent shape
    Ca = int(cfg["audio"]["latent"]["channels"])
    Fa_target = int(cfg["audio"]["latent"]["frames_per_clip"])
    sr = int(cfg["audio"]["sr"])
    fps = int(cfg["video"]["fps"])
    H, W = int(cfg["video"]["size"][0]), int(cfg["video"]["size"][1])

    # ---- Build schedules ----
    betas_v = su.make_beta_schedule(Ttrain_v, kind=cfg["diffusion"]["video"]["schedule"],
                                    min_beta=cfg["diffusion"]["video"]["min_beta"],
                                    max_beta=cfg["diffusion"]["video"]["max_beta"])
    betas_a = su.make_beta_schedule(Ttrain_a, kind=cfg["diffusion"]["audio"]["schedule"],
                                    min_beta=cfg["diffusion"]["audio"]["min_beta"],
                                    max_beta=cfg["diffusion"]["audio"]["max_beta"])
    _, a_bar_v = su.alphas_cumprod_from_betas(betas_v)
    _, a_bar_a = su.alphas_cumprod_from_betas(betas_a)

    sched_v = su.make_sampling_schedule(Ttrain_v, S_v)
    sched_a = su.make_sampling_schedule(Ttrain_a, S_a)

    # ---- Encode prompt latent ----
    B = 1
    if prompt_modality == "video":
        if prompt_video is None:
            raise ValueError("prompt_video frames required for prompt_modality=video")
        # Frames -> latent
        frames = torch.from_numpy(prompt_video).to(device)  # [T,H,W,3] uint8
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2).unsqueeze(0)  # [1,3,T,H,W]
        z_v0 = vid_vae.encode(frames)                     # [1,Cv,T',H',W']
        # Target audio latent init ~ N(0,I)
        z_a = torch.randn(B, Ca, Fa_target, device=device)
        target = "audio"
    else:
        if prompt_audio is None:
            raise ValueError("prompt_audio required for prompt_modality=audio")
        wav = torch.from_numpy(prompt_audio).to(device).view(1, 1, -1)  # [1,1,L]
        z_a0 = aud_codec.encode(wav)                                    # [1,Ca,Fa]
        # Target video latent init ~ N(0,I)
        # Derive T',H',W' from VAE for the given clip length (approximate from fps/down)
        T_in = prompt_video.shape[0] if prompt_video is not None else int(round(cfg["data"]["clip_seconds"] * fps))
        Tp = max(1, T_in // t_down)
        Hp = H // s_down
        Wp = W // s_down
        z_v = torch.randn(B, Cv, Tp, Hp, Wp, device=device)
        target = "video"

    # ---- Prepare token adapters/head ----
    out_dim_v = int(cfg["model"]["heads"]["video"]["out_dim"])
    out_dim_a = int(cfg["model"]["heads"]["audio"]["out_dim"])

    # ---- DDIM loop ----
    # We keep the prompt latent fixed at t=0; only denoise the target.
    if target == "audio":
        # Prompt: z_v0 (fixed), Target evolving: z_a (starts as noise)
        tsteps = sched_a
        guide = g_scale_a
        for i in range(len(tsteps) - 1):
            t_now = tsteps[i].repeat(B).to(device)
            t_prev = tsteps[i + 1].repeat(B).to(device)

            # --- Build tokens for BOTH modalities ---
            tok_v = latents_to_tokens_video(z_v0, t_p=t_p, p=p)   # [B, Nv, out_dim_v]
            tok_a = latents_to_tokens_audio(z_a, l_chunk=l_chunk, s_chunk=s_chunk)  # [B, Na, out_dim_a]

            # Project to width & add timestep emb
            Xv = adapt_v(tok_v)
            Xa = adapt_a(tok_a)
            Xv = add_sinusoidal_timestep(Xv, torch.zeros(B, dtype=torch.long, device=device), tstep_dim)
            Xa = add_sinusoidal_timestep(Xa, t_now, tstep_dim)
            X = torch.cat([Xv, Xa], dim=1)   # [B, Ntot, d]

            # --- CFG: cond vs null (drop the prompt tokens) ---
            # cond: keep prompt video tokens
            h_cond = core(X)
            eps_cond = head({"video": h_cond[:, :Xv.size(1), :], "audio": h_cond[:, Xv.size(1):, :]})["audio"]

            # null: zero-out video tokens (simple null)
            X_null = torch.cat([torch.zeros_like(Xv), Xa], dim=1)
            h_null = core(X_null)
            eps_null = head({"video": h_null[:, :Xv.size(1), :], "audio": h_null[:, Xv.size(1):, :]})["audio"]

            eps_hat_tok = eps_null + guide * (eps_cond - eps_null)  # [B, Na, out_dim_a]

            # Map token eps back to latent shape using overlap-add helper
            eps_hat_lat = tokens_to_latents_audio(eps_hat_tok, Ca=Ca, l_chunk=l_chunk, Fa=Fa_target, stride=s_chunk)

            # DDIM step (audio modality)
            z_a = su.ddim_step(z_a, t_now, t_prev, eps_hat_lat, a_bar_a, eta=eta)

        # Decode audio
        wav_hat = aud_codec.decode(z_a).squeeze(0).squeeze(0).detach().cpu().numpy()
        return {"audio": wav_hat, "sr": sr}

    else:
        # Prompt: z_a0 (fixed), Target evolving: z_v
        tsteps = sched_v
        guide = g_scale_v
        Nv = None
        for i in range(len(tsteps) - 1):
            t_now = tsteps[i].repeat(B).to(device)
            t_prev = tsteps[i + 1].repeat(B).to(device)

            tok_v = latents_to_tokens_video(z_v, t_p=t_p, p=p)    # [B, Nv, out_dim_v]
            tok_a = latents_to_tokens_audio(z_a0, l_chunk=l_chunk, s_chunk=s_chunk)  # [B, Na, out_dim_a]
            Nv = tok_v.size(1)

            Xv = adapt_v(tok_v)
            Xa = adapt_a(tok_a)
            Xv = add_sinusoidal_timestep(Xv, t_now, tstep_dim)
            Xa = add_sinusoidal_timestep(Xa, torch.zeros(B, dtype=torch.long, device=device), tstep_dim)
            X = torch.cat([Xv, Xa], dim=1)

            # CFG: cond keeps audio tokens; null zeros them
            h_cond = core(X)
            eps_cond = head({"video": h_cond[:, :Nv, :], "audio": h_cond[:, Nv:, :]})["video"]

            X_null = torch.cat([Xv, torch.zeros_like(Xa)], dim=1)
            h_null = core(X_null)
            eps_null = head({"video": h_null[:, :Nv, :], "audio": h_null[:, Nv:, :]})["video"]

            eps_hat_tok = eps_null + guide * (eps_cond - eps_null)  # [B, Nv, out_dim_v]

            # Unpatch tokens back to latent-shaped noise
            eps_hat_lat = ops.tube_unpatch_video(
                eps_hat_tok, C=Cv, T=z_v.size(2), H=z_v.size(3), W=z_v.size(4), t=t_p, h=p, w=p
            )

            # DDIM step (video modality)
            z_v = su.ddim_step(z_v, t_now, t_prev, eps_hat_lat, a_bar_v, eta=eta)

        # Decode video frames
        x_hat = vid_vae.decode(z_v).clamp(0, 1)  # [1,3,T,H,W]
        x_hat = (x_hat[0].permute(1, 2, 3, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)  # [T,H,W,3]
        return {"video": x_hat, "fps": fps}


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="One-shot DDIM sampling with CFG (V→A or A→V).")
    ap.add_argument("--config", type=str, nargs="+", required=True, help="One or more YAML configs (merged left→right)")
    ap.add_argument("--frames", type=Path, default=None, help="Prompt: directory of frames (for V→A)")
    ap.add_argument("--audio", type=Path, default=None, help="Prompt: audio file (for A→V)")
    ap.add_argument("--out-frames", type=Path, default=None, help="Output frames directory (for A→V)")
    ap.add_argument("--save-mp4", type=Path, default=None, help="Optional mp4 path (for A→V)")
    ap.add_argument("--out-audio", type=Path, default=None, help="Output wav path (for V→A)")
    ap.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = ap.parse_args()

    cfg = load_config(*args.config)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Build components
    vid_vae, aud_codec, adapt_v, adapt_a, core, head, tstep_dim = build_components(cfg, device)
    load_checkpoint_maybe(cfg, {
        "vid_vae": vid_vae,
        "aud_codec": aud_codec,
        "adapt_v": adapt_v,
        "adapt_a": adapt_a,
        "core": core,
        "head": head,
    })

    prompt_modality = cfg.get("sampling", {}).get("prompt_modality", "video")
    if prompt_modality not in {"video", "audio"}:
        raise ValueError("sampling.prompt_modality must be 'video' or 'audio'")

    prompt_video = None
    prompt_audio = None

    if prompt_modality == "video":
        if args.frames is None:
            raise SystemExit("Provide --frames for prompt_modality=video")
        # Load frames (and resize if needed? assume pre-sized by your pipeline)
        prompt_video = load_frames_dir(args.frames)
        result = sample_one_direction(
            cfg=cfg, vid_vae=vid_vae, aud_codec=aud_codec, adapt_v=adapt_v, adapt_a=adapt_a,
            core=core, head=head, tstep_dim=tstep_dim,
            prompt_modality="video", prompt_video=prompt_video, prompt_audio=None, device=device
        )
        wav = result["audio"]; sr = int(cfg["audio"]["sr"])
        out = args.out_audio or Path("samples_out.wav")
        save_audio_wav(out, wav, sr)
        print(f"[ok] wrote audio → {out}")

    else:
        if args.audio is None:
            raise SystemExit("Provide --audio for prompt_modality=audio")
        prompt_audio = load_audio_wav(args.audio, sr=int(cfg["audio"]["sr"]))
        # If you loaded frames for timing, you can pass them; we derive latent sizes from config otherwise.
        result = sample_one_direction(
            cfg=cfg, vid_vae=vid_vae, aud_codec=aud_codec, adapt_v=adapt_v, adapt_a=adapt_a,
            core=core, head=head, tstep_dim=tstep_dim,
            prompt_modality="audio", prompt_video=None, prompt_audio=prompt_audio, device=device
        )
        frames = result["video"]; fps = result["fps"]
        out_dir = args.out_frames or Path("frames_out")
        write_frames_and_optionally_mp4(frames, out_dir, mp4_path=args.save_mp4, fps=fps)
        print(f"[ok] wrote frames → {out_dir}")
        if args.save_mp4:
            print(f"[ok] wrote mp4 → {args.save_mp4}")

if __name__ == "__main__":
    main()
