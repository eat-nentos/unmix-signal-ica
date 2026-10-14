"""
Compare runs across N.

Reads outputs/N=<n>/separated.npy and the ground-truth sources for each
N present, computes mean and min best-match |corr| for PCA and FastICA,
prints a summary table, saves it as markdown, and plots mean |corr| vs N.
"""
import os
import glob
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def read_sources(path="sources.txt"):
    with open(path) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]


def corr_matrix(A, B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.abs(A @ B.T)


def best_match_scores(S_est, S_true):
    """Return mean and min |corr| over the optimal one-to-one matching."""
    C = corr_matrix(S_est, S_true)
    r, c = linear_sum_assignment(-C)
    scores = C[r, c]
    return float(scores.mean()), float(scores.min())


def main():
    # All source names (for ground-truth loading at each N)
    all_names = read_sources()
    name_to_audio = {nm: f"audio/{nm}.wav" for nm in all_names}

    # Find all N= folders that have results
    n_dirs = sorted(glob.glob("outputs/N=*"))
    rows = []

    for d in n_dirs:
        if not os.path.exists(f"{d}/separated.npy"):
            continue
        n = int(os.path.basename(d).split("=")[1])

        # Names for this N = first n names in sources.txt (order matters!)
        # We stored true sources at mixing time, so load them directly.
        S_true = np.load(f"{d}/true_sources.npy")
        S_ica  = np.load(f"{d}/separated.npy")
        S_pca  = np.load(f"{d}/whitened.npy")

        T = min(S_true.shape[1], S_ica.shape[1], S_pca.shape[1])
        S_true = S_true[:, :T]
        S_ica  = S_ica[:, :T]
        S_pca  = S_pca[:, :T]

        ica_mean, ica_min = best_match_scores(S_ica, S_true)
        pca_mean, pca_min = best_match_scores(S_pca, S_true)

        rows.append((n, pca_mean, pca_min, ica_mean, ica_min))
        print(f"N={n}  PCA mean={pca_mean:.3f} min={pca_min:.3f}  |  "
              f"ICA mean={ica_mean:.3f} min={ica_min:.3f}")

    rows.sort(key=lambda r: r[0])

    # -------- Markdown table --------
    md = "# Separation quality vs N\n\n"
    md += "| N | PCA mean |corr| | PCA min |corr| | FastICA mean |corr| | FastICA min |corr| |\n"
    md += "|---|---|---|---|---|\n"
    for n, pm, pmin, im, imin in rows:
        md += f"| {n} | {pm:.3f} | {pmin:.3f} | {im:.3f} | {imin:.3f} |\n"

    with open("outputs/comparison.md", "w") as f:
        f.write(md)
    print("\nSaved outputs/comparison.md")
    print(md)

    # -------- Plot --------
    ns       = [r[0] for r in rows]
    pca_mean = [r[1] for r in rows]
    ica_mean = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, ica_mean, marker="o", linewidth=2, label="FastICA")
    ax.plot(ns, pca_mean, marker="s", linewidth=2, label="PCA")
    ax.set_xlabel("Number of sources (N)")
    ax.set_ylabel("Mean best-match |corr|")
    ax.set_title("Separation quality vs number of sources")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig("outputs/figures/degradation_vs_N.png", dpi=140)
    plt.close(fig)
    print("Saved outputs/figures/degradation_vs_N.png")


if __name__ == "__main__":
    main()