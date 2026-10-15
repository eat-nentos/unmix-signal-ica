"""
Shared pipeline: mix -> center -> whiten -> FastICA -> evaluate.

One function, run_pipeline(S, A), that the CLI scripts and the Shazam
code can both call. Nothing here touches disk.
"""
import numpy as np

from whiten import center, pca_whiten
from fastica import fastica
from evaluate import corr_matrix, match_estimates_to_sources


def make_mixing_matrix(n, diag=1.0, off=0.5):
    """N x N matrix with 1.0 on the diagonal and 0.5 off it."""
    A = np.full((n, n), off, dtype=np.float32)
    np.fill_diagonal(A, diag)
    return A


def mix(S, A):
    """Linear instantaneous mixture X = A @ S."""
    return A @ S


def run_pipeline(S, A=None, n_restarts=5, seed=0, snr_db=None):
    """
    Run the whole separation pipeline on sources S (shape N x T).

    Parameters
    ----------
    S : np.ndarray, shape (N, T)
        True source signals.
    A : np.ndarray, shape (N, N) or None
        Mixing matrix. If None, uses make_mixing_matrix(N).
    n_restarts : int
        Number of FastICA random restarts.
    seed : int
        Base seed for FastICA.

    Returns
    -------
    dict with keys:
        X        : mixtures (N x T)
        Z        : whitened mixtures
        W_white  : whitening matrix
        W_ica    : unmixing matrix
        S_ica    : ICA-separated signals
        S_pca    : PCA "separated" (the whitened signals)
        metrics  : dict with ica_mean, ica_min, pca_mean, pca_min
    """
    n = S.shape[0]
    if A is None:
        A = make_mixing_matrix(n)

    # 1. Mix
    X = mix(S, A)

    if snr_db is not None:
        rng = np.random.default_rng(seed)
        sig_pow = (X ** 2).mean(axis=1, keepdims=True)
        noise_pow = sig_pow / (10 ** (snr_db / 10))
        X = X + rng.normal(0, np.sqrt(noise_pow), size=X.shape)
        
    # 2. Center + whiten
    Xc, _ = center(X)
    Z, W_white = pca_whiten(Xc)

    # 3. FastICA
    W_ica = fastica(Z, n_restarts=n_restarts, base_seed=seed)
    S_ica = W_ica @ Z

    # 4. PCA output = the whitened signals themselves
    S_pca = Z

    # 5. Evaluate
    C_ica = corr_matrix(S_ica, S)
    C_pca = corr_matrix(S_pca, S)
    match_ica, _ = match_estimates_to_sources(C_ica)
    match_pca, _ = match_estimates_to_sources(C_pca)
    ica_scores = [s for _, _, s in match_ica]
    pca_scores = [s for _, _, s in match_pca]

    return {
        "X": X,
        "Z": Z,
        "W_white": W_white,
        "W_ica": W_ica,
        "S_ica": S_ica,
        "S_pca": S_pca,
        "metrics": {
            "ica_mean": float(np.mean(ica_scores)),
            "ica_min":  float(np.min(ica_scores)),
            "pca_mean": float(np.mean(pca_scores)),
            "pca_min":  float(np.min(pca_scores)),
        },
    }