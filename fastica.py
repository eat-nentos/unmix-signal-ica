"""
FastICA from scratch with N components and multiple random restarts.

Loads the whitened signals saved by whiten.py, runs symmetric-decorrelation
FastICA with tanh nonlinearity from several random inits, keeps the run
with the highest non-Gaussianity objective, and saves the unmixing matrix
and recovered sources.
"""
import os
import glob
import numpy as np
import soundfile as sf

def read_sources(path="sources.txt"):
    with open(path) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]
    
# ---------- Nonlinearity ----------
def g(u):
    return np.tanh(u)

def g_prime(u):
    return 1.0 - np.tanh(u) ** 2


# ---------- Symmetric decorrelation ----------
def symmetric_decorrelation(W):
    """Return (W W^T)^(-1/2) W, making rows orthonormal."""
    WWt = W @ W.T
    ev, V = np.linalg.eigh(WWt)
    ev = np.clip(ev, 1e-12, None)
    return np.diag(1.0 / np.sqrt(ev)) @ V.T @ W


# ---------- FastICA core ----------
def log_cosh(x):
    return np.logaddexp(x, -x) - np.log(2.0)


def fastica_once(Z, seed, max_iter=2000, tol=1e-5, mu=1.0):
    """One FastICA run with a damped fixed-point update."""
    n, N = Z.shape
    rng = np.random.default_rng(seed)
    W = symmetric_decorrelation(rng.standard_normal((n, n)))

    for it in range(max_iter):
        W_old = W
        WZ = W @ Z
        G  = np.tanh(WZ)
        Gp = 1.0 - G ** 2
        W_update = (G @ Z.T) / N - Gp.mean(axis=1, keepdims=True) * W
        W = symmetric_decorrelation(W_update)

        delta = np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1.0))
        if delta < tol:
            return W, it + 1
    return W, max_iter


def non_gaussianity_score(Z, W):
    """Sum of mean(log cosh(w^T z)) across components. Higher = more non-Gaussian."""
    WZ = W @ Z
    return float(np.mean(log_cosh(WZ)))

#---------------------Multiple Restarts---------------------------------
def fastica(Z, n_restarts=5, base_seed=0):
    best_W, best_score, best_iters, best_seed = None, -np.inf, None, None
    for k in range(n_restarts):
        seed = base_seed + k
        W, iters = fastica_once(Z, seed)
        score = non_gaussianity_score(Z, W)
        print(f"  restart {k+1}: seed={seed}  iters={iters}  score={score:.4f}")
        if score > best_score:
            best_W, best_score, best_iters, best_seed = W, score, iters, seed
    print(f"  Best restart: seed={best_seed}  iters={best_iters}  score={best_score:.4f}")
    return best_W

# ---------- Objective: how non-Gaussian are the components? ----------
def non_gaussianity_score(Z, W):
    """
    Sum of |kurtosis| across components.

    Non-Gaussian sources (speech, music, rain) have high |kurtosis|;
    Gaussian noise has near-zero kurtosis. Higher score = more non-Gaussian
    = better separation.
    """
    WZ = W @ Z
    mu = WZ.mean(axis=1, keepdims=True)
    var = ((WZ - mu) ** 2).mean(axis=1, keepdims=True)
    fourth = ((WZ - mu) ** 4).mean(axis=1, keepdims=True)
    kurt = fourth / (var ** 2 + 1e-12) - 3.0
    return float(np.sum(np.abs(kurt)))

def main():
    # Find latest N= directory
    names = read_sources()
    n = len(names)
    out_dir = f"outputs/N={n}"
    if not os.path.exists(f"{out_dir}/whitened.npy"):
        raise FileNotFoundError(f"No whitened signals at {out_dir}. Run whiten.py first.")
    Z = np.load(f"{out_dir}/whitened.npy")
    n, N = Z.shape
    print(f"Using {out_dir}  (n={n})")

    print("\nRunning FastICA with 5 random restarts...")
    W_ica = fastica(Z, n_restarts=5)

    S_est = W_ica @ Z
    print("\nRecovered sources shape:", S_est.shape)

    # Normalize for export
    S_audio = S_est / (np.max(np.abs(S_est), axis=1, keepdims=True) + 1e-12) * 0.95

    for i in range(n):
        sf.write(f"{out_dir}/separated_{i+1}.wav", S_audio[i], 16000)

    np.save(f"{out_dir}/separated.npy", S_est)
    np.save(f"{out_dir}/unmixing_matrix.npy", W_ica)

    print(f"\nSaved to {out_dir}/:")
    for i in range(n):
        print(f"  separated_{i+1}.wav")
    print("  separated.npy")
    print("  unmixing_matrix.npy")


if __name__ == "__main__":
    main()