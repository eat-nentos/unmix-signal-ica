"""
snr_sweep.py - separation quality vs input SNR, using the real pipeline.
Noise is added inside pipeline.run_pipeline, before centering, whitening and FastICA.
N is read from sources.txt.
"""
import contextlib, io, os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import run_pipeline

SNRS = [30, 20, 10, 5, 0]      # dB, clean -> noisy
SEEDS = [0, 1, 2]
SECONDS, SR = 12, 16000


def read_sources(path="sources.txt"):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def main():
    names = read_sources()
    n = len(names)
    S = np.vstack([sf.read(f"audio/{nm}.wav")[0] for nm in names])[:, : SECONDS * SR]
    print(f"N = {n}, sources = {names}")

    ica = np.zeros((len(SNRS), len(SEEDS)))
    pca = np.zeros((len(SNRS), len(SEEDS)))
    for i, snr in enumerate(SNRS):
        for j, seed in enumerate(SEEDS):
            with contextlib.redirect_stdout(io.StringIO()):   # silence FastICA prints
                r = run_pipeline(S, n_restarts=5, seed=seed, snr_db=snr)
            ica[i, j] = r["metrics"]["ica_mean"]
            pca[i, j] = r["metrics"]["pca_mean"]

    os.makedirs("outputs/figures", exist_ok=True)
    md = f"# SNR sweep at N={n} - mean over {len(SEEDS)} seeds\n\n"
    md += "| SNR (dB) | FastICA mean \\|corr\\| | PCA mean \\|corr\\| |\n|---|---|---|\n"
    for i, snr in enumerate(SNRS):
        md += (f"| {snr} | {ica[i].mean():.3f} +/- {ica[i].std():.3f} | "
               f"{pca[i].mean():.3f} +/- {pca[i].std():.3f} |\n")
    print(md)
    with open("outputs/snr_sweep_results.md", "w", encoding="utf-8") as f:
        f.write(md)

    plt.figure(figsize=(7, 4))
    plt.errorbar(SNRS, ica.mean(1), yerr=ica.std(1), marker="o", capsize=4, label="FastICA")
    plt.errorbar(SNRS, pca.mean(1), yerr=pca.std(1), marker="s", capsize=4, label="PCA")
    plt.gca().invert_xaxis()
    plt.xlabel("Input SNR (dB), higher is cleaner")
    plt.ylabel("Mean matched |corr|")
    plt.title(f"Separation quality vs noise (N={n})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/snr_sweep.png", dpi=140)
    plt.close()
    print("Saved outputs/snr_sweep_results.md and outputs/figures/snr_sweep.png")


if __name__ == "__main__":
    main()