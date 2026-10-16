"""
convergence_plots.py - FastICA convergence diagnostics (the 'loss graph').

FastICA has no training loss, so we plot its two convergence signals for
every random restart:
  1. delta per iteration  (max | |diag(W W_old^T)| - 1 |, log scale);
     the run stops when delta < tol.
  2. the restart-selection objective (sum of |excess kurtosis|) per iteration.

It re-runs the same update as fastica.fastica_once (same seeds, same maths)
but records the history. Usage: python convergence_plots.py [N ...]
Also writes a text log. Works for every outputs/N=<n>/ that has whitened.npy.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastica import symmetric_decorrelation, non_gaussianity_score

N_RESTARTS = 5
BASE_SEED = 0
MAX_ITER = 2000
TOL = 1e-5
SCORE_EVERY = 10      # objective is costly, so sample it every 10 iterations


def traced_run(Z, seed):
    """Same update as fastica_once, but returns (W, iters, deltas, (score_its, scores))."""
    n, N = Z.shape
    rng = np.random.default_rng(seed)
    W = symmetric_decorrelation(rng.standard_normal((n, n)))
    deltas, scores, score_its = [], [], []
    for it in range(MAX_ITER):
        W_old = W
        G = np.tanh(W @ Z)
        Gp = 1.0 - G ** 2
        W = symmetric_decorrelation((G @ Z.T) / N - Gp.mean(axis=1, keepdims=True) * W)
        delta = np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1.0))
        deltas.append(float(delta))
        done = delta < TOL or it == MAX_ITER - 1
        if it % SCORE_EVERY == 0 or done:
            scores.append(non_gaussianity_score(Z, W))
            score_its.append(it + 1)
        if delta < TOL:
            return W, it + 1, deltas, (score_its, scores)
    return W, MAX_ITER, deltas, (score_its, scores)


def run_for(n_dir):
    n = int(os.path.basename(n_dir).split("=")[1])
    Z = np.load(f"{n_dir}/whitened.npy")
    runs = []
    for k in range(N_RESTARTS):
        seed = BASE_SEED + k
        W, iters, deltas, (score_its, scores) = traced_run(Z, seed)
        runs.append(dict(seed=seed, iters=iters, deltas=deltas, score_its=score_its,
                         scores=scores, final=scores[-1], converged=deltas[-1] < TOL))
    best = max(range(len(runs)), key=lambda i: runs[i]["final"])

    # ---- log ----
    os.makedirs("outputs/logs", exist_ok=True)
    lines = [f"FastICA convergence log, N={n}, tol={TOL}, max_iter={MAX_ITER}", "",
             "restart  seed  iters  converged  final_score(sum|kurt|)"]
    for k, r in enumerate(runs):
        tag = "  <- best" if k == best else ""
        lines.append(f"{k+1:>7}  {r['seed']:>4}  {r['iters']:>5}  {str(r['converged']):>9}  {r['final']:.4f}{tag}")
    log = "\n".join(lines)
    with open(f"outputs/logs/convergence_N{n}.txt", "w", encoding="utf-8") as f:
        f.write(log + "\n")
    print(log + "\n")

    # ---- figure ----
    os.makedirs(f"{n_dir}/figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, r in enumerate(runs):
        lw = 2.2 if k == best else 1.1
        lab = f"restart {k+1} (seed {r['seed']}, {r['iters']} it)" + (" - best" if k == best else "")
        ax[0].semilogy(range(1, len(r["deltas"]) + 1), r["deltas"], lw=lw, label=lab)
        ax[1].plot(r["score_its"], r["scores"], lw=lw, label=lab)
    ax[0].axhline(TOL, color="gray", ls="--", lw=1, label=f"tol = {TOL:g}")
    ax[0].set_xlabel("Iteration"); ax[0].set_ylabel("delta (log scale)")
    ax[0].set_title(f"Convergence of W (N={n})")
    ax[1].set_xlabel("Iteration"); ax[1].set_ylabel("Sum of |excess kurtosis|")
    ax[1].set_title("Selection objective per iteration")
    for a in ax:
        a.grid(True, alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    out = f"{n_dir}/figures/convergence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out} and outputs/logs/convergence_N{n}.txt\n")


def main():
    import sys
    wanted = {int(a) for a in sys.argv[1:]}          # optional: python convergence_plots.py 2 3
    dirs = sorted(d for d in glob.glob("outputs/N=*") if os.path.exists(f"{d}/whitened.npy")
                  and (not wanted or int(os.path.basename(d).split("=")[1]) in wanted))
    if not dirs:
        raise FileNotFoundError("No outputs/N=*/whitened.npy found. Run whiten.py first.")
    for d in dirs:
        run_for(d)


if __name__ == "__main__":
    main()