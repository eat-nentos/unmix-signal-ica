"""
shazam_weight_sweep_noise.py  (E3 with noise)

Same as shazam_weight_sweep.py but adds Gaussian noise to the mixtures
at 20 dB SNR. This makes the result harder to dismiss ("too perfect to
be convincing" without noise).

Runs 3 seeds per cell and reports mean ± std.
"""
import os
import pickle
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import run_pipeline, make_mixing_matrix
from fingerprint import identify_with_confidence

ANCHOR_NAME = "anchor_track"
WEIGHTS = [1.0, 0.5, 0.25, 0.1, 0.05]
SNR_DB = 20
SEEDS = [0, 1, 2]


def read_sources(path="sources.txt"):
    with open(path) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]


def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-12)


def best_match(signals, db):
    best_result = None
    for x in signals:
        best, score, conf, _ = identify_with_confidence(normalize(x), db)
        if best_result is None or score > best_result[1]:
            best_result = (best, score, conf)
    return best_result


def add_noise(X, snr_db, rng):
    """Add Gaussian noise to each row of X at the given SNR (dB)."""
    sig_pow = (X ** 2).mean(axis=1, keepdims=True)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    return X + rng.normal(0, np.sqrt(noise_pow), size=X.shape)


def main():
    names = read_sources()
    n = len(names)
    music_idx = names.index("music")
    print(f"N = {n}, sources = {names}")
    print(f"Music source index: {music_idx}")
    print(f"SNR: {SNR_DB} dB | Seeds: {SEEDS}")

    S = np.vstack([sf.read(f"audio/{nm}.wav")[0] for nm in names])
    S = S[:, :12 * 16000]

    print("\nLoading fingerprint DB...")
    with open("outputs/fingerprint_db.pkl", "rb") as f:
        data = pickle.load(f)
    db = data["db"]
    print(f"  {len(db)} hashes")

    rows = []
    print(f"\n{'weight':>7s}  {'type':>5s}  {'top1':>6s}  "
          f"{'win score (mean ± std)':>26s}  {'best decoy (mean)':>18s}")
    print("-" * 80)

    for w in WEIGHTS:
        # Collect per-seed scores for mix and ica
        mix_scores, ica_scores = [], []
        mix_top1, ica_top1 = [], []

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            A = make_mixing_matrix(n)
            A[:, music_idx] *= w

            result = run_pipeline(S, A, n_restarts=5, seed=seed)
            X = add_noise(result["X"], SNR_DB, rng)
            S_ica = result["S_ica"]

            # mix
            m_best, m_score, m_conf = best_match(X, db)
            mix_scores.append(m_score)
            mix_top1.append(1 if m_best == ANCHOR_NAME else 0)

            # ica
            i_best, i_score, i_conf = best_match(S_ica, db)
            ica_scores.append(i_score)
            ica_top1.append(1 if i_best == ANCHOR_NAME else 0)

        rows.append({
            "weight": w,
            "mix_top1": float(np.mean(mix_top1)),
            "mix_score_mean": float(np.mean(mix_scores)),
            "mix_score_std": float(np.std(mix_scores)),
            "ica_top1": float(np.mean(ica_top1)),
            "ica_score_mean": float(np.mean(ica_scores)),
            "ica_score_std": float(np.std(ica_scores)),
        })

        print(f"{w:>7.3f}  {'mix':>5s}  {np.mean(mix_top1):>6.2f}  "
              f"{np.mean(mix_scores):>10.1f} ± {np.std(mix_scores):<10.1f}  {'-':>18s}")
        print(f"{'':>7s}  {'ica':>5s}  {np.mean(ica_top1):>6.2f}  "
              f"{np.mean(ica_scores):>10.1f} ± {np.std(ica_scores):<10.1f}  {'-':>18s}")

    # ---- Markdown table ----
    md = f"# E3 with noise — {SNR_DB} dB SNR, mean over {len(SEEDS)} seeds\n\n"
    md += "| Music weight | Mix top-1 | Mix score | ICA top-1 | ICA score |\n"
    md += "|---|---|---|---|---|\n"
    for r in rows:
        md += (f"| {r['weight']:.2f} | "
               f"{r['mix_top1']:.2f} | "
               f"{r['mix_score_mean']:.1f} ± {r['mix_score_std']:.1f} | "
               f"{r['ica_top1']:.2f} | "
               f"{r['ica_score_mean']:.1f} ± {r['ica_score_std']:.1f} |\n")
    os.makedirs("outputs/figures", exist_ok=True)
    with open("outputs/e3_noise_results.md", "w") as f:
        f.write(md)
    print(f"\nSaved outputs/e3_noise_results.md")
    print(md)

    # ---- Plot ----
    ws = [r["weight"] for r in rows]
    mix_scores = [r["mix_score_mean"] for r in rows]
    ica_scores = [r["ica_score_mean"] for r in rows]
    mix_std = [r["mix_score_std"] for r in rows]
    ica_std = [r["ica_score_std"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ws, mix_scores, yerr=mix_std, marker="o", capsize=4, label="Raw mixture")
    ax.errorbar(ws, ica_scores, yerr=ica_std, marker="s", capsize=4, label="ICA output")
    ax.set_xscale("log")
    ax.set_xlabel("Music weight in mixing matrix A (log scale)")
    ax.set_ylabel(f"Fingerprint score (mean over {len(SEEDS)} seeds)")
    ax.set_title(f"E3 with {SNR_DB} dB noise")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/figures/e3_weight_sweep_noise.png", dpi=140)
    plt.close(fig)
    print("Saved outputs/figures/e3_weight_sweep_noise.png")


if __name__ == "__main__":
    main()