"""
Build the fingerprint database from library/wav/*.wav and save it as
outputs/fingerprint_db.pkl.

Run this once; reload with pickle.load in identify scripts.
"""
import os
import glob
import pickle
import soundfile as sf

from fingerprint import build_db


def load_library(folder="library/wav"):
    files = sorted(glob.glob(f"{folder}/*.wav"))
    if not files:
        raise FileNotFoundError(
            f"No WAV files in {folder}/. Run convert_library.py first."
        )
    songs = {}
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        x, _ = sf.read(p)
        songs[name] = x
    return songs


def main():
    print("Loading library...")
    songs = load_library()
    print(f"  {len(songs)} songs loaded: {list(songs.keys())}")

    print("Building fingerprint database...")
    db = build_db(songs)
    print(f"  {len(db)} unique hashes")

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/fingerprint_db.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"db": db, "song_names": list(songs.keys())}, f)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nSaved {out_path}  ({size_mb:.1f} MB)")
    print("Reload with:")
    print("    import pickle")
    print(f"    data = pickle.load(open('{out_path}', 'rb'))")
    print("    db = data['db']")


if __name__ == "__main__":
    main()