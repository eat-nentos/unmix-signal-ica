"""
shazam_weight_sweep.py  (Experiment E3)

Progressively bury the music in the mixture by scaling the music column
of the mixing matrix A. At each weight, run the full pipeline and
fingerprint both the raw mixture and the ICA output.

Shows where the raw mixture stops matching the anchor song while the
ICA output still does — the definitive demonstration that separation
improves fingerprinting.
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

ANCHOR_NAME = "anchor_track"      # rename if your library file differs
WEIGHTS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
SEED = 0


def read_sources(path="sources.txt"):
    with open(path) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]


def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-12)


def best_match(signals, db):
    results = []
    for x in signals:
        best, score, conf, _ = identify_with_confidence(normalize(x), db)
        results.append((best, score, conf))
    return max(results, key=lambda r: r[1])


def main():
    names = read_sources()
    n = len(names)
    music_idx = names.index("music")
    print(f"N = {n}, sources = {names}")
    print(f"Music is source index {music_idx} ('{names[music_idx]}')")

    S = np.vstack([sf.read(f"audio/{nm}.wav")[0] for nm in names])
    T = min(S.shape[1], 12 * 16000)
    S = S[:, :T]

    print("\nLoading fingerprint DB...")
    with open("outputs/fingerprint_db.pkl", "rb") as f:
        data = pickle.load(f)
    db = data["db"]
    print(f"  {len(db)} hashes, library: {data['song_names']}")

    rows = []
    print(f"\n{'weight':>7s}  {'mix match':22s}  {'mix score':>9s}  {'mix conf':>8s}  "
          f"{'ica match':22s}  {'ica score':>9s}  {'ica conf':>8s}")
    print("-" * 110)

    for w in WEIGHTS:
        A = make_mixing_matrix(n)
        A[:, music_idx] *= w

        result = run_pipeline(S, A, n_restarts=5, seed=SEED)
        X = result["X"]
        S_ica = result["S_ica"]

        mix_best, mix_score, mix_conf = best_match(X, db)
        ica_best, ica_score, ica_conf = best_match(S_ica, db)

        rows.append((w, mix_best, mix_score, mix_conf,
                     ica_best, ica_score, ica_conf))

        print(f"{w:>7.3f}  {str(mix_best):22s}  {mix_score:>9d}  {mix_conf:>8.2f}  "
              f"{str(ica_best):22s}  {ica_score:>9d}  {ica_conf:>8.2f}")

    md = "# E3 — Fingerprint quality vs music weight in A\n\n"
    md += "| Music weight | Mix best | Mix score | Mix conf | "
    md += "ICA best | ICA score | ICA conf |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for w, mb, ms, mc, ib, isc, ic in rows:
        md += (f"| {w:.3f} | {mb} | {ms} | {mc:.2f} | "
               f"{ib} | {isc} | {ic:.2f} |\n")
    os.makedirs("outputs/figures", exist_ok=True)
    with open("outputs/e3_results.md", "w") as f:
        f.write(md)
    print("\nSaved outputs/e3_results.md")

    ws = [r[0] for r in rows]
    mix_scores = [r[2] for r in rows]
    ica_scores = [r[5] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ws, mix_scores, marker="o", label="Raw mixture")
    ax.plot(ws, ica_scores, marker="s", label="ICA output")
    ax.set_xscale("log")
    ax.set_xlabel("Music weight in mixing matrix A (log scale)")
    ax.set_ylabel("Fingerprint score for anchor track")
    ax.set_title("E3: Fingerprint score vs how buried the music is")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/figures/e3_weight_sweep.png", dpi=140)
    plt.close(fig)
    print("Saved outputs/figures/e3_weight_sweep.png")

    print("\n=== Interpretation ===")
    mix_breaks = None
    ica_breaks = None
    for w, mb, ms, mc, ib, isc, ic in rows:
        if mix_breaks is None and mb != ANCHOR_NAME:
            mix_breaks = w
        if ica_breaks is None and ib != ANCHOR_NAME:
            ica_breaks = w

    if mix_breaks is None:
        print(f"  Raw mixture matched the anchor at ALL weights tested.")
    else:
        print(f"  Raw mixture stops matching the anchor below weight {mix_breaks:.3f}")

    if ica_breaks is None:
        print(f"  ICA output matched the anchor at ALL weights tested.")
    else:
        print(f"  ICA output stops matching the anchor below weight {ica_breaks:.3f}")

    if mix_breaks is not None and ica_breaks is not None:
        if ica_breaks < mix_breaks:
            print(f"\n  → ICA extends identification down to weight {ica_breaks:.3f}, "
                  f"while the raw mixture breaks at {mix_breaks:.3f}. "
                  f"That's a {mix_breaks/ica_breaks:.1f}x improvement.")


if __name__ == "__main__":
    main()