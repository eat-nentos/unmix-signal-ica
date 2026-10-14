import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/figures", exist_ok=True)

sr = 16000
seeds = [0, 1, 2]              # average over 3 seeds
snrs  = [30, 20, 10, 5, 0]     # dB, clean -> noisy

# ---------- Load sources ----------
speech, _ = sf.read("audio/speech.mp3")
music,  _ = sf.read("audio/music.mp3")
n = min(len(speech), len(music))
speech, music = speech[:n], music[:n]

S = np.vstack([speech, music])               # (2, N)
A = np.array([[1.0, 0.6], [0.5, 1.0]])       # same mixing matrix as before


# ---------- Whitening (same as whiten.py) ----------
def whiten(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    C = (Xc @ Xc.T) / Xc.shape[1]
    eigvals, E = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals, E = eigvals[order], E[:, order]
    W = np.diag(1 / np.sqrt(eigvals)) @ E.T
    return W @ Xc, W


# ---------- Symmetric decorrelation (same as fastica.py) ----------
def sym_decorr(W):
    WWt = W @ W.T
    ev, V = np.linalg.eigh(WWt)
    ev = np.clip(ev, 1e-12, None)
    return np.diag(1 / np.sqrt(ev)) @ V.T @ W


# ---------- FastICA (same algorithm) ----------
def fastica(Z, seed=0, max_iter=500, tol=1e-6):
    rng = np.random.default_rng(seed)
    W = sym_decorr(rng.standard_normal((2, 2)))
    for _ in range(max_iter):
        W_old = W
        WZ = W @ Z
        G, Gp = np.tanh(WZ), 1 - np.tanh(WZ) ** 2
        W = sym_decorr((G @ Z.T) / Z.shape[1] - Gp.mean(axis=1, keepdims=True) * W)
        if np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1)) < tol:
            break
    return W


# ---------- Matching-based recovery score ----------
def recovery_corr(S_est, S_true):
    A_ = S_est - S_est.mean(axis=1, keepdims=True)
    B_ = S_true - S_true.mean(axis=1, keepdims=True)
    A_ /= np.linalg.norm(A_, axis=1, keepdims=True) + 1e-12
    B_ /= np.linalg.norm(B_, axis=1, keepdims=True) + 1e-12
    C = np.abs(A_ @ B_.T)
    # Greedy best assignment for 2x2: try both pairings, take the better average
    score_a = (C[0, 0] + C[1, 1]) / 2
    score_b = (C[0, 1] + C[1, 0]) / 2
    return max(score_a, score_b)


# ---------- Main loop ----------
results = np.zeros((len(snrs), len(seeds)))

for i, snr_db in enumerate(snrs):
    for j, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)

        X = A @ S                                        # clean mixtures

        # Add Gaussian noise at the desired SNR, per row
        sig_pow   = (X ** 2).mean(axis=1, keepdims=True)
        noise_pow = sig_pow / (10 ** (snr_db / 10))
        Xn = X + rng.normal(0, np.sqrt(noise_pow), size=X.shape)

        Z, _ = whiten(Xn)
        W    = fastica(Z, seed=seed)
        S_est = W @ Z

        results[i, j] = recovery_corr(S_est, S)

mean = results.mean(axis=1)
std  = results.std(axis=1)

print("SNR (dB) | mean |corr| over seeds")
for s, m, d in zip(snrs, mean, std):
    print(f"  {s:3d}    |  {m:.3f} ± {d:.3f}")


# ---------- Plot ----------
plt.figure(figsize=(7, 4))
plt.errorbar(snrs, mean, yerr=std, marker="o", capsize=4)
plt.gca().invert_xaxis()             # higher SNR on the left
plt.xlabel("SNR (dB) — higher is cleaner")
plt.ylabel("Mean |corr| (matched)")
plt.title("FastICA separation quality vs SNR")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/figures/snr_sweep.png", dpi=140)
plt.close()
print("\nSaved outputs/figures/snr_sweep.png")