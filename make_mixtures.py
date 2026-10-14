"""
Create mixtures from N sources.

Reads sources.txt, loads audio/<name>.wav for each, stacks them into an
(N, T) matrix S, applies an (N, N) mixing matrix A, and saves
outputs/mix1.wav ... outputs/mixN.wav.
"""
import os
import numpy as np
import soundfile as sf


def read_sources(path="sources.txt"):
    with open(path) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]


def build_mixing_matrix(n):
    """
    Build an N x N mixing matrix.

    Diagonal entries are 1.0 (each source appears in its own mixture).
    Off-diagonal entries are 0.5 (each source leaks into every other
    mixture at half strength). This keeps the matrix well-conditioned
    so no single source dominates.
    """
    A = np.full((n, n), 0.5, dtype=np.float32)
    np.fill_diagonal(A, 1.0)
    return A


def main():
    names = read_sources()
    n = len(names)
    print(f"Creating {n} mixtures from sources: {names}")

    # Load each source
    signals = []
    sr = None
    for name in names:
        x, s = sf.read(f"audio/{name}.wav")
        if sr is None:
            sr = s
        elif s != sr:
            raise ValueError(f"Sample rate mismatch in {name}.wav")
        signals.append(x)

    # Trim all to the same length
    length = min(len(x) for x in signals)
    S = np.vstack([x[:length] for x in signals])       # shape (N, T)
    print(f"Source matrix S shape: {S.shape}  (sample rate {sr})")

    # Mix: X = A @ S
    A = build_mixing_matrix(n)
    X = A @ S                                          # shape (N, T)
    print("Mixing matrix A:")
    print(A)

    # Normalise each mixture row to avoid clipping
    X = X / (np.max(np.abs(X), axis=1, keepdims=True) + 1e-12) * 0.95

    # Save mixtures
    out_dir = f"outputs/N={n}"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        out = f"{out_dir}/mix{i+1}.wav"
        sf.write(out, X[i], sr)
        print(f"  saved {out}")

    # Save A and S for later comparison
    np.save(f"{out_dir}/mixing_matrix.npy", A)
    np.save(f"{out_dir}/true_sources.npy", S)
    print(f"Saved {out_dir}/mixing_matrix.npy and {out_dir}/true_sources.npy")


if __name__ == "__main__":
    main()