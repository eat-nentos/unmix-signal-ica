"""
convergence_diagnose.py - does FastICA really settle? (no early stopping)
Usage: python convergence_diagnose.py [N] [seed] [iters]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from fastica import symmetric_decorrelation
from evaluate import corr_matrix, match_estimates_to_sources

N_SRC = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
TOL = 1e-5
EVERY = 5

d = f"outputs/N={N_SRC}"
Z = np.load(f"{d}/whitened.npy")
T = np.load(f"{d}/true_sources.npy")
n, N = Z.shape


def matched(W):
    m, _ = match_estimates_to_sources(corr_matrix(W @ Z, T))
    s = [x[2] for x in m]
    return float(np.mean(s)), float(np.min(s))


W = symmetric_decorrelation(np.random.default_rng(SEED).standard_normal((n, n)))
delta, pdelta, c_it, c_mean, c_min = [], [], [], [], []
for it in range(ITERS):
    W_old = W
    G = np.tanh(W @ Z)
    W = symmetric_decorrelation((G @ Z.T) / N - (1 - G ** 2).mean(axis=1, keepdims=True) * W)
    M = np.abs(W @ W_old.T)
    delta.append(float(np.max(np.abs(np.diag(M) - 1))))
    r, c = linear_sum_assignment(-M)
    pdelta.append(float(np.max(np.abs(M[r, c] - 1))))
    if it % EVERY == 0:
        mean, mn = matched(W)
        c_it.append(it + 1); c_mean.append(mean); c_min.append(mn)

delta, pdelta = np.array(delta), np.array(pdelta)
first = int(np.argmax(delta < TOL)) + 1 if (delta < TOL).any() else None
print(f"N={N_SRC} seed={SEED} iterations={ITERS}")
print(f"first iteration with delta < {TOL:g}: {first}")
print(f"median delta: {np.median(delta):.2e} | median perm-invariant delta: {np.median(pdelta):.2e}")
print(f"smallest perm-invariant delta: {pdelta.min():.2e}")
print(f"matched |corr| over run: min {min(c_min):.3f}, max {max(c_mean):.3f}, last {c_mean[-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(delta, lw=0.6, label="delta (as in fastica.py)")
ax[0].semilogy(pdelta, lw=0.9, label="permutation-invariant delta")
ax[0].axhline(TOL, color="gray", ls="--", lw=1, label=f"tol = {TOL:g}")
ax[0].set_xlabel("Iteration"); ax[0].set_ylabel("delta (log)"); ax[0].legend(fontsize=7)
ax[1].plot(c_it, c_mean, label="mean matched |corr|")
ax[1].plot(c_it, c_min, label="min matched |corr|")
ax[1].set_xlabel("Iteration"); ax[1].set_ylabel("|corr| with true sources"); ax[1].legend(fontsize=7)
for a in ax:
    a.grid(True, alpha=0.3)
fig.suptitle(f"FastICA without early stopping (N={N_SRC}, seed {SEED})")
fig.tight_layout()
out = f"{d}/figures/convergence_diagnostic.png"
fig.savefig(out, dpi=140)
print("Saved", out)