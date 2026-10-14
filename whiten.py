"""
Center and whiten N mixtures.

Importable API:
    center(X)       -> Xc, mean
    pca_whiten(Xc)  -> Z, W

CLI usage (unchanged):
    python whiten.py
"""
import os
import numpy as np
import soundfile as sf


def read_sources(path="sources.txt"):
    with open(path) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]


def center(X):
    """
    Subtract the per-row mean from X (shape N x T).
    Returns:
        Xc   : centered signals, same shape
        mean : the per-row means that were subtracted, shape (N, 1)
    """
    mean = X.mean(axis=1, keepdims=True)
    return X - mean, mean


def pca_whiten(Xc):
    """
    PCA-whiten centered signals Xc (shape N x T).

    Returns:
        Z : whitened signals (N x T), covariance == identity
        W : the whitening matrix (N x N) such that Z = W @ Xc
    """
    N = Xc.shape[1]

    # Covariance
    C = (Xc @ Xc.T) / N

    # Eigendecomposition (symmetric)
    eigvals, E = np.linalg.eigh(C)

    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals, E = eigvals[order], E[:, order]

    # Clip tiny eigenvalues for numerical safety
    eigvals = np.clip(eigvals, 1e-12, None)

    # Whitening matrix W = diag(1/sqrt(lambda)) @ E.T
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    W = D_inv_sqrt @ E.T

    Z = W @ Xc
    return Z, W


def main():
    names = read_sources()
    n = len(names)
    out_dir = f"outputs/N={n}"
    if not os.path.exists(f"{out_dir}/mix1.wav"):
        raise FileNotFoundError(
            f"No mixtures at {out_dir}. Run make_mixtures.py first."
        )
    print(f"Using {out_dir} with n = {n}")

    # Load mixtures
    X = []
    sr = None
    for i in range(1, n + 1):
        x, s = sf.read(f"{out_dir}/mix{i}.wav")
        if sr is None:
            sr = s
        X.append(x)
    X = np.vstack(X)
    print("Input shape:", X.shape)

    # Center
    Xc, mean = center(X)
    print("Mean before:", mean.ravel())
    print("Mean after :", Xc.mean(axis=1))

    # Whiten
    Z, W = pca_whiten(Xc)

    # Verify
    Cz = (Z @ Z.T) / Z.shape[1]
    print("\nCovariance of whitened signals (should be ~identity):")
    print(np.round(Cz, 4))

    # Save
    np.save(f"{out_dir}/whitened.npy", Z)
    np.save(f"{out_dir}/whitening_matrix.npy", W)
    np.save(f"{out_dir}/centered_mixtures.npy", Xc)
    print(f"\nSaved to {out_dir}/:")
    print("  whitened.npy")
    print("  whitening_matrix.npy")
    print("  centered_mixtures.npy")


if __name__ == "__main__":
    main()