"""
Generate figures for the current N run.

Reads the latest outputs/N=*/ and produces:
  - waveforms.png        N rows x 4 cols: sources, mixtures, whitened, ICA
  - spectrograms.png     N rows x 3 cols: sources, mixtures, ICA
  - correlation_heatmaps.png   PCA vs FastICA, N x N each
"""
import os
import glob
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def main():
    names = read_sources()
    n = len(names)
    out_dir = f"outputs/N={n}"
    if not os.path.exists(f"{out_dir}/separated.npy"):
        raise FileNotFoundError(
            f"No results found at {out_dir}. Run the pipeline at N={n} first."
        )
    fig_dir = f"{out_dir}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Plotting for {out_dir}  (n={n})")

    sr = 16000

    # Load everything
    S_true = np.vstack([sf.read(f"audio/{nm}.wav")[0] for nm in names])
    S_mix  = np.vstack([sf.read(f"{out_dir}/mix{i+1}.wav")[0] for i in range(n)])
    Z      = np.load(f"{out_dir}/whitened.npy")
    S_ica  = np.load(f"{out_dir}/separated.npy")

    T = min(S_true.shape[1], S_mix.shape[1], Z.shape[1], S_ica.shape[1])
    S_true = S_true[:, :T]
    S_mix  = S_mix[:, :T]
    Z      = Z[:, :T]
    S_ica  = S_ica[:, :T]

    t = np.arange(T) / sr

    # ---------- Figure 1: waveform grid ----------
    fig, axes = plt.subplots(n, 4, figsize=(16, 2.2 * n), sharex=True)
    if n == 1:
        axes = axes.reshape(1, -1)
    cols = ["Original (true)", "Mixture", "Whitened", "ICA separated"]
    for j in range(4):
        axes[0, j].set_title(cols[j], fontsize=11)
    for i in range(n):
        axes[i, 0].plot(t, S_true[i], linewidth=0.5, color="C0")
        axes[i, 1].plot(t, S_mix[i],  linewidth=0.5, color="C1")
        axes[i, 2].plot(t, Z[i],      linewidth=0.5, color="C2")
        axes[i, 3].plot(t, S_ica[i],  linewidth=0.5, color="C3")
        for j in range(4):
            axes[i, j].set_ylabel(names[i][:10] if j == 0 else "")
            axes[i, j].grid(True, alpha=0.2)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    axes[-1, 2].set_xlabel("time (s)")
    axes[-1, 3].set_xlabel("time (s)")
    fig.suptitle(f"Sources -> mixtures -> whitened -> separated (N={n})", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/waveforms.png", dpi=130)
    plt.close(fig)
    print(f"  saved {fig_dir}/waveforms.png")

    # ---------- Figure 2: spectrograms ----------
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.2 * n), sharex=True, sharey=True)
    if n == 1:
        axes = axes.reshape(1, -1)
    cols = ["Source", "Mixture", "ICA separated"]
    for j in range(3):
        axes[0, j].set_title(cols[j], fontsize=11)
    for i in range(n):
        axes[i, 0].specgram(S_true[i], NFFT=1024, Fs=sr, noverlap=512, cmap="magma")
        axes[i, 1].specgram(S_mix[i],  NFFT=1024, Fs=sr, noverlap=512, cmap="magma")
        axes[i, 2].specgram(S_ica[i],  NFFT=1024, Fs=sr, noverlap=512, cmap="magma")
        axes[i, 0].set_ylabel(names[i][:10])
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    axes[-1, 2].set_xlabel("time (s)")
    fig.suptitle(f"Spectrograms (N={n})", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/spectrograms.png", dpi=130)
    plt.close(fig)
    print(f"  saved {fig_dir}/spectrograms.png")

    # ---------- Figure 3: correlation heatmaps ----------
    C_ica = corr_matrix(S_ica, S_true)
    C_pca = corr_matrix(Z, S_true)

    fig, axes = plt.subplots(1, 2, figsize=(4 * n, 3.5 * n / 2 + 2))
    for ax, C, title in zip(axes, [C_pca, C_ica], ["PCA components", "FastICA (tanh)"]):
        im = ax.imshow(C, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels([f"est{i+1}" for i in range(n)])
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                        color="white" if C[i,j] < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="|corr|")
    fig.suptitle(f"|corr| estimates vs sources (N={n})", fontsize=13)
    fig.savefig(f"{fig_dir}/correlation_heatmaps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_dir}/correlation_heatmaps.png")


if __name__ == "__main__":
    main()