"""
shazam_weight_sweep_noise.py  (E3 with noise — corrected)

Adds Gaussian noise at 20 dB SNR to the MIXTURES, BEFORE whitening and
FastICA, so both the raw-mixture branch and the ICA branch see the same
noisy data. Reports mean ± std over 3 seeds.

"Noise" is added inside pipeline.run_pipeline(snr_db=...) — this script
no longer touches noise directly.
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
WEIGHTS = [1.0, 0.5, 0.25, 0.1, 0.02]
SNR_DB = 20
SEEDS = [0, 1, 2]
SCORE_THRESHOLD = 10   # scores below this are random collisions


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


def main():
    names = read_sources()
    n = len(names)
    music_idx = names.index("music")
    print(f"N = {n}, sources = {names}")
    print(f"Music source index: {music_idx}")
    print(f"SNR: {SNR_DB} dB | Seeds: {SEEDS} | Score threshold: {SCORE_THRESHOLD}")

    S = np.vstack([sf.read(f"audio/{nm}.wav")[0] for nm in names])
    S = S[:, :12 * 16000]

    print("\nLoading fingerprint DB...")
    with open("outputs/fingerprint_db.pkl", "rb") as f:
        data = pickle.load(f)
    db = data["db"]
    print(f"  {len(db)} hashes")

    rows = []
    print(f"\n{'weight':>7s}  {'type':>5s}  {'top1':>6s}  "
          f"{'win score (mean ± std)':>24s}  {'ICA |corr|':>12s}")
    print("-" * 75)

    for w in WEIGHTS:
        mix_scores, ica_scores = [], []
        mix_top1, ica_top1 = [], []
        ica_corrs = []

        for seed in SEEDS:
            A = make_mixing_matrix(n)
            A[:, music_idx] *= w

            result = run_pipeline(S, A, n_restarts=5, seed=seed, snr_db=SNR_DB)
            X = result["X"]                  # already noisy
            S_ica = result["S_ica"]          # computed from noisy data

            # Which ICA component contains the music?
            # Use the one with highest |corr| to the true music source.
            true_music = S[music_idx]
            tm = true_music - true_music.mean()
            tm = tm / (np.linalg.norm(tm) + 1e-12)
            corrs = []
            for j in range(n):
                e = S_ica[j] - S_ica[j].mean()
                e = e / (np.linalg.norm(e) + 1e-12)
                corrs.append(abs(np.dot(e, tm)))
            ica_corrs.append(float(np.max(corrs)))

            # Mix branch
            m_best, m_score, m_conf = best_match(X, db)
            mix_scores.append(m_score)
            mix_top1.append(1 if (m_best == ANCHOR_NAME and
                                  m_score >= SCORE_THRESHOLD) else 0)

            # ICA branch
            i_best, i_score, i_conf = best_match(S_ica, db)
            ica_scores.append(i_score)
            ica_top1.append(1 if (i_best == ANCHOR_NAME and
                                  i_score >= SCORE_THRESHOLD) else 0)

        rows.append({
            "weight": w,
            "mix_top1": float(np.mean(mix_top1)),
            "mix_score_mean": float(np.mean(mix_scores)),
            "mix_score_std": float(np.std(mix_scores)),
            "ica_top1": float(np.mean(ica_top1)),
            "ica_score_mean": float(np.mean(ica_scores)),
            "ica_score_std": float(np.std(ica_scores)),
            "ica_corr_mean": float(np.mean(ica_corrs)),
            "ica_corr_std": float(np.std(ica_corrs)),
        })

        print(f"{w:>7.3f}  {'mix':>5s}  {np.mean(mix_top1):>6.2f}  "
              f"{np.mean(mix_scores):>10.1f} ± {np.std(mix_scores):<10.1f}  {'-':>12s}")
        print(f"{'':>7s}  {'ica':>5s}  {np.mean(ica_top1):>6.2f}  "
              f"{np.mean(ica_scores):>10.1f} ± {np.std(ica_scores):<10.1f}  "
              f"{np.mean(ica_corrs):>5.3f} ± {np.std(ica_corrs):.3f}")

    # ---- Markdown table ----
    md = f"# E3 with noise — {SNR_DB} dB SNR, mean over {len(SEEDS)} seeds\n\n"
    md += f"*Score threshold for 'identified': rank-1 AND score >= {SCORE_THRESHOLD}.*\n\n"
    md += "| Music weight | Mix top-1 | Mix score | ICA top-1 | ICA score | ICA \\|corr\\| to music |\n"
    md += "|---|---|---|---|---|---|\n"
    for r in rows:
        md += (f"| {r['weight']:.2f} | "
               f"{r['mix_top1']:.2f} | "
               f"{r['mix_score_mean']:.1f} ± {r['mix_score_std']:.1f} | "
               f"{r['ica_top1']:.2f} | "
               f"{r['ica_score_mean']:.1f} ± {r['ica_score_std']:.1f} | "
               f"{r['ica_corr_mean']:.3f} ± {r['ica_corr_std']:.3f} |\n")
    os.makedirs("outputs/figures", exist_ok=True)
    with open("outputs/e3_noise_results.md", "w", encoding="utf-8") as f:
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
    ax.axhline(SCORE_THRESHOLD, color="gray", linestyle="--", linewidth=1,
               label=f"threshold ({SCORE_THRESHOLD})")
    ax.set_xscale("log")
    ax.set_xlabel("Music weight in mixing matrix A (log scale)")
    ax.set_ylabel(f"Fingerprint score (mean over {len(SEEDS)} seeds)")
    ax.set_title(f"E3 with {SNR_DB} dB noise (corrected: noise before ICA)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/figures/e3_weight_sweep_noise.png", dpi=140)
    plt.close(fig)
    print("Saved outputs/figures/e3_weight_sweep_noise.png")


if __name__ == "__main__":
    main()