#!/usr/bin/env python3
import json
from pathlib import Path

VID_ROOT = Path("data/video/GRID/frames_tmp")
AUD_ROOT = Path("data/audio/GRID/wav16k")
OUT_JSON = Path("data/GRID/clips.json")

def find_wav(spk: str, utt: str) -> Path | None:
    # Try common layouts:
    cands = [
        AUD_ROOT / spk / f"{utt}.wav",        # data/audio/GRID/wav16k/s2/utt.wav
        AUD_ROOT / spk / spk / f"{utt}.wav",  # data/audio/GRID/wav16k/s2/s2/utt.wav
    ]
    for p in cands:
        if p.exists():
            return p
    # Fallback: glob search
    hits = list(AUD_ROOT.rglob(f"{utt}.wav"))
    return hits[0] if hits else None

def main():
    clips = []
    for spk_dir in sorted(VID_ROOT.glob("s*")):
        spk = spk_dir.name  # e.g., s2
        for utt_dir in sorted(spk_dir.iterdir()):
            if not utt_dir.is_dir(): 
                continue
            utt = utt_dir.name  # e.g., sbwh6s
            clip0 = utt_dir / "clips" / "clip_0000"
            frames_dir = clip0 if clip0.exists() else (utt_dir / "frames")
            if not frames_dir.exists():
                continue
            wav = find_wav(spk, utt)
            if wav is None:
                # Skip if no matching audio found
                continue
            clips.append({
                "video_frames_dir": str(frames_dir),
                "audio_wav_path": str(wav),
                "fps": 16,
                "sr": 16000,
                "clip_seconds": 3.0
            })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"clips": clips}, f, indent=2)
    print(f"Wrote {len(clips)} clips to {OUT_JSON}")

if __name__ == "__main__":
    main()
