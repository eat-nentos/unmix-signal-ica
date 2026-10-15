"""
Convert library/*.{mp3,wav,m4a,flac,ogg,aac} to 16 kHz mono WAV.

Output: library/wav/<same_name>.wav
Also trims to a max of 60 seconds to keep the fingerprint database small.
"""
import os
import glob
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

TARGET_SR = 16000
MAX_SECONDS = 60.0
IN_EXTS = ["mp3", "wav", "m4a", "flac", "ogg", "aac"]


def convert(in_path, out_path):
    x, sr = sf.read(in_path)

    # Stereo -> mono
    if x.ndim > 1:
        x = x.mean(axis=1)

    # Resample
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, sr // g)
        sr = TARGET_SR

    # Trim to MAX_SECONDS
    max_n = int(MAX_SECONDS * sr)
    if len(x) > max_n:
        x = x[:max_n]

    # Normalise
    x = x / (np.max(np.abs(x)) + 1e-12) * 0.95

    sf.write(out_path, x, sr)
    return len(x) / sr


def main():
    os.makedirs("library/wav", exist_ok=True)

    # Find every input file in library/ (not in library/wav/)
    files = []
    for ext in IN_EXTS:
        files.extend(glob.glob(f"library/*.{ext}"))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError("No audio files found in library/")

    print(f"Converting {len(files)} library files to 16 kHz mono WAV...\n")
    for in_path in files:
        name = os.path.splitext(os.path.basename(in_path))[0]
        out_path = f"library/wav/{name}.wav"
        dur = convert(in_path, out_path)
        print(f"  {os.path.basename(in_path):30s} -> {out_path}  ({dur:.1f}s)")

    print(f"\nDone. {len(files)} files converted.")


if __name__ == "__main__":
    main()