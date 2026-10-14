"""
shazam_on_ica.py

Run the mini-Shazam fingerprinter on the ICA pipeline outputs.

For a given N, loads:
  - the fingerprint database (outputs/fingerprint_db.pkl)
  - the mixture, PCA, and ICA outputs (outputs/N=<n>/)

Runs identify() on each and prints:
  - which library song was matched
  - the score and confidence ratio

Then picks which ICA output is the music by highest fingerprint score
for the anchor track — WITHOUT using ground truth.
"""
import os
import pickle
import numpy as np
import soundfile as sf

from fingerprint import identify_with_confidence

# --- Which library song corresponds to the "music" source? ---
# Change this if your anchor file has a different name in library/wav/
ANCHOR_NAME = "anchor_track"


def read_sources(path="sources.txt"):
    with open(path) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]


def load_db(path="outputs/fingerprint_db.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["db"], data["song_names"]


def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-12)


def run_batch(signals, prefix, db):
    """Run identify() on each signal, return list of (idx, best, score, conf)."""
    rows = []
    for i, sig in enumerate(signals):
        best, score, conf, _ = identify_with_confidence(normalize(sig), db)
        rows.append((i, best, score, conf))
        print(f"  {f'{prefix}_{i+1}':8s}  {str(best):22s}  {score:6d}  {conf:6.2f}")
    return rows


def main():
    names = read_sources()
    n = len(names)
    out_dir = f"outputs/N={n}"
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"No outputs at {out_dir}. Run the pipeline first.")

    print(f"N = {n}, sources = {names}")
    print(f"Loading fingerprint DB...")
    db, lib_names = load_db()
    print(f"  Library: {lib_names}")
    print(f"  Hashes:  {len(db)}")

    # Load pipeline outputs
    mixtures = [sf.read(f"{out_dir}/mix{i+1}.wav")[0] for i in range(n)]
    pca      = np.load(f"{out_dir}/whitened.npy")
    ica      = np.load(f"{out_dir}/separated.npy")

    print(f"\n=== Fingerprinting each output ===")
    header = f"  {'output':8s}  {'best match':22s}  {'score':>6s}  {'conf':>6s}"
    sep    = f"  {'-'*8}  {'-'*22}  {'-'*6}  {'-'*6}"

    print("\n[Mixtures]")
    print(header); print(sep)
    mix_rows = run_batch(mixtures, "mix", db)

    print("\n[PCA components]")
    print(header); print(sep)
    pca_rows = run_batch(pca, "pca", db)

    print("\n[ICA components]")
    print(header); print(sep)
    ica_rows = run_batch(ica, "ica", db)

    # ---- Decision: which ICA component is the music? ----
    # Rule: among ICA components, pick the one whose best match is the
    # anchor track AND has the highest score. No ground truth used.
    print(f"\n=== Fingerprint-based decision (no ground truth) ===")
    anchor_hits = [(i, s, c) for i, b, s, c in ica_rows if b == ANCHOR_NAME]
    if anchor_hits:
        best_i, best_score, best_conf = max(anchor_hits, key=lambda r: r[1])
        print(f"  ICA component #{best_i+1} matched '{ANCHOR_NAME}' "
              f"with score {best_score} (conf {best_conf:.2f})")
        print(f"  → We declare ICA component #{best_i+1} is the music.")
    else:
        print(f"  No ICA component matched '{ANCHOR_NAME}'.")
        print(f"  → Fingerprinter cannot identify the music. Possible causes:")
        print(f"      - ANCHOR_NAME is wrong (current: '{ANCHOR_NAME}')")
        print(f"      - ICA didn't isolate the music cleanly at this N")
        print(f"      - Music weight in A is too low for the fingerprinter")

    # ---- Summary table ----
    print(f"\n=== Summary ===")
    print(header); print(sep)
    for i, b, s, c in mix_rows:
        tag = " ← music?" if b == ANCHOR_NAME else ""
        print(f"  {'mix_'+str(i+1):8s}  {str(b):22s}  {s:6d}  {c:6.2f}{tag}")
    for i, b, s, c in pca_rows:
        tag = " ← music?" if b == ANCHOR_NAME else ""
        print(f"  {'pca_'+str(i+1):8s}  {str(b):22s}  {s:6d}  {c:6.2f}{tag}")
    for i, b, s, c in ica_rows:
        tag = " ← music?" if b == ANCHOR_NAME else ""
        print(f"  {'ica_'+str(i+1):8s}  {str(b):22s}  {s:6d}  {c:6.2f}{tag}")


if __name__ == "__main__":
    main()