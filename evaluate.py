"""
Evaluate ICA and PCA separations at any N.

Loads the true sources, ICA estimates, and PCA components (the whitened
signals) for the most recent N= directory, computes N x N absolute
correlation matrices, solves the permutation ambiguity with the Hungarian
algorithm (scipy.optimize.linear_sum_assignment), and reports per-source
best-match correlations plus a summary table.
"""
import os
import glob
import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment


def read_sources(path="sources.txt"):
    with open(path) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]


def corr_matrix(A, B):
    """N x M absolute Pearson correlation between rows of A and rows of B."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.abs(A @ B.T)


def match_estimates_to_sources(C):
    """
    Solve the optimal one-to-one assignment between estimates (rows) and
    sources (columns) maximizing total |corr|.

    Returns: list of (estimate_idx, source_idx, |corr|) and total score.
    """
    row_ind, col_ind = linear_sum_assignment(-C)   # minimize -C = maximize C
    matches = [(r, c, C[r, c]) for r, c in zip(row_ind, col_ind)]
    total = C[row_ind, col_ind].sum()
    return matches, total


def main():
    # Find latest N= run
    names = read_sources()
    n = len(names)
    out_dir = f"outputs/N={n}"
    if not os.path.exists(f"{out_dir}/separated.npy"):
        raise FileNotFoundError(f"No results at {out_dir}. Run fastica.py first.")
    print(f"Evaluating {out_dir}  (n={n})")

    # Load ground truth
    S_true = np.vstack([sf.read(f"audio/{nm}.wav")[0] for nm in names])
    # Load estimates
    S_ica = np.load(f"{out_dir}/separated.npy")
    S_pca = np.load(f"{out_dir}/whitened.npy")

    # Trim to common length
    T = min(S_true.shape[1], S_ica.shape[1], S_pca.shape[1])
    S_true = S_true[:, :T]
    S_ica  = S_ica[:, :T]
    S_pca  = S_pca[:, :T]

    # Correlations
    C_ica = corr_matrix(S_ica, S_true)
    C_pca = corr_matrix(S_pca, S_true)

    # Optimal matching
    match_ica, _ = match_estimates_to_sources(C_ica)
    match_pca, _ = match_estimates_to_sources(C_pca)

    def report(label, C, matches):
        print(f"\n=== {label} ===")
        print("|corr| matrix  (rows = estimates, cols = sources)")
        header = "        " + "  ".join(f"{nm:>8s}" for nm in names)
        print(header)
        for i in range(n):
            row = "  ".join(f"{C[i,j]:8.3f}" for j in range(n))
            print(f"  est{i+1}: {row}")

        print(f"\nOptimal matching:")
        scores = []
        for est_idx, src_idx, s in sorted(matches, key=lambda x: x[1]):
            print(f"  {names[src_idx]:8s}  <-  est{est_idx+1:2d}   |corr| = {s:.3f}")
            scores.append(s)
        print(f"  Mean |corr| = {np.mean(scores):.3f}")
        print(f"  Min  |corr| = {np.min(scores):.3f}")
        return scores

    ica_scores = report("FastICA", C_ica, match_ica)
    pca_scores = report("PCA (whitened signals)", C_pca, match_pca)

    # Save markdown table
    md = f"# Results at N={n}\n\n"
    md += "| Source | PCA \\|corr\\| | FastICA \\|corr\\| |\n"
    md += "|---|---|---|\n"
    pca_by_src = {s: c for _, s, c in match_pca}
    ica_by_src = {s: c for _, s, c in match_ica}
    for j, nm in enumerate(names):
        md += f"| {nm} | {pca_by_src.get(j, 0):.3f} | {ica_by_src.get(j, 0):.3f} |\n"
    md += f"| **Mean** | **{np.mean(pca_scores):.3f}** | **{np.mean(ica_scores):.3f}** |\n"
    md += f"| **Min**  | **{np.min(pca_scores):.3f}** | **{np.min(ica_scores):.3f}** |\n"

    with open(f"{out_dir}/results.md", "w") as f:
        f.write(md)
    print(f"\nSaved {out_dir}/results.md")
    print("\n" + md)


if __name__ == "__main__":
    main()