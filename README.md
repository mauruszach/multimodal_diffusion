An audio-video cross-modal diffusion model.

multimodal_diffusion/
├─ README.md
├─ pyproject.toml            # or setup.cfg/requirements.txt
├─ configs/
│  ├─ mvp.yaml               # model/dataset/training defaults (3s, 16fps, 16kHz)
│  ├─ a2v.yaml               # force Audio→Video at inference defaults
│  └─ v2a.yaml               # force Video→Audio at inference defaults
├─ scripts/
│  ├─ extract_frames.py      # (you already have this)
│  └─ preprocess_audio.py    # (you already have this)
├─ data/
│  └─ (your processed datasets live here: */clips/*, clips.json)
├─ avdiff/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ io.py               # safe load/save, JSON/YAML helpers
│  │  ├─ ops.py              # small tensor ops, chunk helpers
│  │  └─ schedule_utils.py   # cosine/linear beta schedules, alpha_bar, etc.
│  ├─ datasets/
│  │  ├─ av_manifest.py      # loads aligned (video, audio) clip manifests
│  │  ├─ frames_dataset.py   # returns video frame tensors [B,3,T,H,W]
│  │  └─ audio_dataset.py    # returns audio waveform/log-mel [B, C, L] or [B, n_mels, F]
│  ├─ models/
│  │  ├─ encoders/
│  │  │  ├─ vae_video3d.py   # 3D VAE: encode video → z_v [B,Cv',T',H',W'] / decode back
│  │  │  └─ audio_codec.py   # audio encoder/decoder: wav↔z_a [B,Ca',Fa] (frame_hop_ms aware)
│  │  ├─ tokenizers.py       # tube_patch_video(), chunk_1d() for audio; returns tokens
│  │  ├─ adapters.py         # linear P_v, P_a to width d; modality/type/time embeddings
│  │  ├─ mmdt.py             # Multimodal Diffusion Transformer (shared denoiser core)
│  │  ├─ heads/
│  │  │  └─ noise_heads.py   # MultiModalNoiseHead (video/audio output dims per token)
│  │  └─ schedules.py        # per-modality {betas, alphas, q_sample, ddim_step}
│  ├─ train/
│  │  ├─ collate.py          # builds batch dicts, pads, selects target (V or A)
│  │  ├─ losses.py           # noise MSE (targets only); optional alignment loss
│  │  ├─ mask_schedule.py    # Bernoulli schedule for {'video'} vs {'audio'} target
│  │  ├─ trainer.py          # training loop (DDP-friendly), EMA, logging, ckpts
│  │  └─ train_joint.py      # entrypoint: loads config, builds model, trains any→any
│  ├─ infer/
│  │  ├─ sample_clip.py      # one-shot DDIM + CFG: prompt=video ⇒ audio, or prompt=audio ⇒ video
│  │  └─ stream_infer.py     # sliding-window long streams (warm-start + crossfade)
│  └─ eval/
│     ├─ av_sync.py          # simple audio/video sync proxy metrics
│     ├─ audio_quality.py    # mel cepstral / PESQ-style hooks (optional)
│     └─ video_metrics.py    # basic frame stats; LPIPS hook (optional)
└─ tests/
   ├─ test_shapes.py         # quick shape/smoke tests for modules
   └─ smoke_train.py         # tiny synthetic run (overfit a few batches)
