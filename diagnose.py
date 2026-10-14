import numpy as np
import soundfile as sf

names = ["speech", "music", "rain", "traffic"]
d = "outputs/N=4"

S = np.load(f"{d}/separated.npy")            # (4, T)
orig = np.vstack([sf.read(f"audio/{n}.wav")[0] for n in names])

n = min(S.shape[1], orig.shape[1])
S, orig = S[:, :n], orig[:, :n]

C = np.abs(np.corrcoef(np.vstack([S, orig]))[:4, 4:])
print("|corr| matrix  (rows = estimates, cols = speech, music, rain, traffic)")
print(np.round(C, 3))