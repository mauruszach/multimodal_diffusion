#!/usr/bin/env python3
"""
audio_quality.py — simple audio quality metrics & hooks.

Metrics implemented:
  - SDR/SNR-like:        10 * log10(||ref||^2 / ||ref - est||^2)
  - Log-mel L1:          mean absolute error in log-mel space
  - Spectral convergence: ||S_est - S_ref||_F / ||S_ref||_F
  - MCD (Mel Cepstral Dist.): ~6.14185 * RMSE between MFCCs, with optional DTW alignment

Optional (if packages available):
  - PESQ (wb/nb) via `pesq` package
  - STOI via `pystoi`

CLI:
  python -m avdiff.eval.audio_quality --ref ref.wav --est est.wav --sr 16000
"""

from __future__ import annotations
import argparse
from typing import Dict, Tuple

import numpy as np

try:
    import librosa
except Exception:
    librosa = None

# Optional metrics
try:
    from pesq import pesq  # type: ignore
except Exception:
    pesq = None

try:
    from pystoi import stoi  # type: ignore
except Exception:
    stoi = None


# -------------- Core metrics --------------

def load_audio(path: str, sr: int) -> np.ndarray:
    if librosa is None:
        raise RuntimeError("librosa required. `pip install librosa`")
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)

def snr_like(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref.astype(np.float32)
    est = est.astype(np.float32)
    minL = min(len(ref), len(est))
    ref = ref[:minL]; est = est[:minL]
    num = np.sum(ref ** 2) + 1e-10
    den = np.sum((ref - est) ** 2) + 1e-10
    return float(10.0 * np.log10(num / den))

def logmel_l1(ref: np.ndarray, est: np.ndarray, sr: int,
              n_mels: int = 64, n_fft: int = 1024, hop_length: int = 256) -> float:
    if librosa is None:
        raise RuntimeError("librosa required for spectrograms.")
    def _logmel(y):
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                           hop_length=hop_length, n_mels=n_mels,
                                           fmin=20.0, fmax=sr/2, power=2.0)
        return np.log(S + 1e-6).astype(np.float32)  # [M, T]
    A = _logmel(ref)
    B = _logmel(est)
    T = min(A.shape[1], B.shape[1])
    return float(np.mean(np.abs(A[:, :T] - B[:, :T])))

def spectral_convergence(ref: np.ndarray, est: np.ndarray, sr: int,
                         n_fft: int = 1024, hop_length: int = 256) -> float:
    if librosa is None:
        raise RuntimeError("librosa required for spectrograms.")
    S_ref = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop_length))
    S_est = np.abs(librosa.stft(est, n_fft=n_fft, hop_length=hop_length))
    T = min(S_ref.shape[1], S_est.shape[1])
    num = np.linalg.norm(S_est[:, :T] - S_ref[:, :T], ord="fro")
    den = np.linalg.norm(S_ref[:, :T], ord="fro") + 1e-10
    return float(num / den)

def mcd(ref: np.ndarray, est: np.ndarray, sr: int,
        n_mfcc: int = 13, hop_length: int = 256, use_dtw: bool = True) -> float:
    """
    Mel Cepstral Dist. (lower is better), in dB.
    MCD ≈ 6.14185 * sqrt( Σ_{d=1..K} (mc_ref[d] - mc_est[d])^2 ), averaged over frames.
    We exclude c0 by default (librosa's mfcc has c0..c_{n_mfcc-1}; we slice 1:).
    With use_dtw=True, we align frames using DTW on MFCC space.
    """
    if librosa is None:
        raise RuntimeError("librosa required for MFCC/DTW.")
    ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).astype(np.float32)
    est_mfcc = librosa.feature.mfcc(y=est, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).astype(np.float32)
    # exclude c0
    R = ref_mfcc[1:, :].T  # [Tr, K-1]
    E = est_mfcc[1:, :].T  # [Te, K-1]
    if use_dtw:
        D, wp = librosa.sequence.dtw(X=R.T, Y=E.T, metric="euclidean")  # warping path
        pairs = np.array(wp)[::-1]  # in ascending time
        Rs = R[pairs[:, 0]]
        Es = E[pairs[:, 1]]
    else:
        T = min(R.shape[0], E.shape[0])
        Rs, Es = R[:T], E[:T]
    diff = Rs - Es
    rmse = np.sqrt(np.sum(diff ** 2, axis=1) + 1e-9)  # per-frame
    mcd_const = 10.0 / np.log(10.0) * np.sqrt(2.0)   # ≈ 6.14185
    return float(mcd_const * np.mean(rmse))

# -------------- Optional metrics --------------

def pesq_score(ref: np.ndarray, est: np.ndarray, sr: int) -> float | None:
    """
    PESQ via ITU-T P.862 wrapper. Returns None if unavailable or unsupported sr.
    16k: 'wb', 8k: 'nb'.
    """
    if pesq is None:
        return None
    if sr == 16000:
        mode = "wb"
    elif sr == 8000:
        mode = "nb"
    else:
        return None
    try:
        return float(pesq(sr, ref, est, mode))
    except Exception:
        return None

def stoi_score(ref: np.ndarray, est: np.ndarray, sr: int) -> float | None:
    if stoi is None:
        return None
    try:
        return float(stoi(ref, est, sr, extended=False))
    except Exception:
        return None


# -------------- Batch metric --------------

def evaluate_pair(ref_wav: str, est_wav: str, sr: int = 16000) -> Dict[str, float | None]:
    ref = load_audio(ref_wav, sr=sr)
    est = load_audio(est_wav, sr=sr)
    return {
        "snr": snr_like(ref, est),
        "logmel_l1": logmel_l1(ref, est, sr=sr),
        "spec_conv": spectral_convergence(ref, est, sr=sr),
        "mcd": mcd(ref, est, sr=sr),
        "pesq": pesq_score(ref, est, sr=sr),
        "stoi": stoi_score(ref, est, sr=sr),
    }


# -------------------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser(description="Audio quality metrics for a reference vs estimate.")
    ap.add_argument("--ref", type=str, required=True, help="Reference wav path")
    ap.add_argument("--est", type=str, required=True, help="Estimated wav path")
    ap.add_argument("--sr", type=int, default=16000, help="Sample rate for loading")
    args = ap.parse_args()

    scores = evaluate_pair(args.ref, args.est, sr=args.sr)
    for k, v in scores.items():
        print(f"{k:10s}: {('%.4f' % v) if v is not None else 'N/A'}")

if __name__ == "__main__":
    main()
