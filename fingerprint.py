"""
fingerprint.py — mini-Shazam

Pipeline:
    audio -> spectrogram -> peaks -> paired hashes -> database
    query -> same -> offset voting -> best matching song

Importable API:
    get_peaks(x, sr)
    get_hashes(peaks)
    build_db(songs)          # songs: dict {name: signal}
    identify(x, db)          # returns [(song, score), ...] best first
    identify_with_confidence(x, db)
"""
import numpy as np
from collections import defaultdict, Counter
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter


# ============================================================
# 1. Spectrogram peaks
# ============================================================
def get_peaks(x, sr=16000, nperseg=512, nbhd=(15, 15), pct=90):
    """
    Return a list of (time_frame, freq_bin) peaks.

    A point is a peak if:
      - it's the local maximum in a (nbhd[0] x nbhd[1]) neighbourhood, AND
      - its log-magnitude is above the pct-th percentile of the whole
        spectrogram.
    """
    f, t, S = spectrogram(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    S = np.log(S + 1e-10)

    is_peak = (S == maximum_filter(S, size=nbhd)) & (S > np.percentile(S, pct))
    fi, ti = np.nonzero(is_peak)

    order = np.argsort(ti)
    return list(zip(ti[order], fi[order]))            # (t_frame, freq_bin)


# ============================================================
# 2. Pair hashes
# ============================================================
def get_hashes(peaks, fan_out=5, dt_min=1, dt_max=64):
    """
    For each anchor peak, pair it with up to `fan_out` next peaks
    within a time window [dt_min, dt_max].

    Returns a list of (hash, anchor_time), where
    hash = (freq1, freq2, dt) — all integers.
    """
    peaks = sorted(peaks)                              # by (time, freq)
    hashes = []
    for i, (t1, f1) in enumerate(peaks):
        for j in range(i + 1, len(peaks)):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt > dt_max:
                break
            if dt >= dt_min:
                hashes.append(((int(f1), int(f2), int(dt)), int(t1)))
            # stop after `fan_out` successful pairs
            if j - i >= fan_out:
                break
    return hashes


# ============================================================
# 3. Build database
# ============================================================
def build_db(songs):
    """
    songs: dict {name: signal}
    Returns: dict {hash: [(name, anchor_time), ...]}
    """
    db = defaultdict(list)
    for name, x in songs.items():
        peaks = get_peaks(x)
        for h, t in get_hashes(peaks):
            db[h].append((name, t))
    return db


# ============================================================
# 4. Identify a query
# ============================================================
def identify(x, db):
    """
    Match a query signal against the database.
    Returns: list of (song_name, score) sorted best-first.

    Each hash hit votes for (song, t_db - t_query). Only the true song
    produces many hits at the same offset, because the query was cut
    from one specific position in the original.
    """
    peaks = get_peaks(x)
    query_hashes = get_hashes(peaks)

    votes = defaultdict(Counter)                       # votes[song][offset]
    for h, tq in query_hashes:
        for name, tdb in db.get(h, []):
            votes[name][tdb - tq] += 1

    if not votes:
        return []

    scores = {name: max(c.values()) for name, c in votes.items()}
    return sorted(scores.items(), key=lambda kv: -kv[1])


def identify_with_confidence(x, db):
    """
    Returns: (best_song, best_score, confidence_ratio, all_scores)
    confidence_ratio = best_score / second_best_score (1 if only one match).
    """
    scores = identify(x, db)
    if not scores:
        return None, 0, 0.0, []
    best_name, best_score = scores[0]
    second = scores[1][1] if len(scores) > 1 else 0
    ratio = best_score / max(second, 1)
    return best_name, best_score, ratio, scores


# ============================================================
# 5. Smoke test: build DB and identify a clip from each song
# ============================================================
if __name__ == "__main__":
    import glob, os, soundfile as sf

    print("Loading library...")
    files = sorted(glob.glob("library/wav/*.wav"))
    if not files:
        raise SystemExit("No files in library/wav/. Run convert_library.py first.")
    songs = {os.path.splitext(os.path.basename(p))[0]: sf.read(p)[0]
             for p in files}
    print(f"  {len(songs)} songs loaded")

    print("Building fingerprint database...")
    db = build_db(songs)
    print(f"  {len(db)} unique hashes")

    print("\nSelf-test: 5-second clip from the middle of each song")
    print(f"  {'query':22s}  {'best match':22s}  score  conf  result")
    print(f"  {'-'*22}  {'-'*22}  -----  ----  ------")
    for name, x in songs.items():
        start = len(x) // 2
        clip = x[start:start + 5 * 16000]
        best, score, conf, all_scores = identify_with_confidence(clip, db)
        ok = "OK" if best == name else "FAIL"
        print(f"  {name:22s}  {str(best):22s}  {score:5d}  {conf:4.2f}  {ok}")