"""
Prepare audio sources.

Reads sources.txt (one name per line), finds raw_<name>.* in the project
root, converts each to mono 16 kHz WAV, trims to 12 s, normalises, and
saves as audio/<name>.wav.
"""
import os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

TARGET_SR = 16000
DURATION = 12.0
EXTS = ["wav", "mp3", "m4a", "flac", "ogg", "aac"]


def read_sources(path="sources.txt"):
    """Read source names from sources.txt, ignoring blanks and comments."""
    with open(path) as f:
        names = [line.strip() for line in f
                 if line.strip() and not line.startswith("#")]
    if len(names) < 2:
        raise ValueError("Need at least 2 sources in sources.txt")
    return names


def find_raw(prefix):
    """Find raw_<prefix>.<ext> in the project root."""
    for ext in EXTS:
        p = f"raw_{prefix}.{ext}"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No raw_{prefix}.* found in project root "
        f"(looked for one of: {', '.join(EXTS)})"
    )


def convert(in_path, out_path):
    x, sr = sf.read(in_path)

    # Stereo -> mono
    if x.ndim > 1:
        x = x.mean(axis=1)

    # Resample to TARGET_SR
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, sr // g)
        sr = TARGET_SR

    # Trim to exactly DURATION seconds
    n = int(DURATION * sr)
    if len(x) < n:
        raise ValueError(
            f"{in_path} is too short: {len(x)/sr:.2f}s, need {DURATION}s"
        )
    x = x[:n]

    # Normalise to avoid clipping
    x = x / (np.max(np.abs(x)) + 1e-12) * 0.95

    os.makedirs("audio", exist_ok=True)
    sf.write(out_path, x, sr)
    print(f"  {in_path:20s} -> {out_path}  ({len(x)/sr:.2f}s @ {sr}Hz)")


def main():
    names = read_sources()
    print(f"Preparing {len(names)} sources: {names}")
    for name in names:
        raw = find_raw(name)
        convert(raw, f"audio/{name}.wav")
    print("Done.")


if __name__ == "__main__":
    main()