"""Split MDM's results.npy into one joints .npy per (prompt, repetition).

MDM stores motion as (reps * prompts, 22, 3, T), repetition-major.
"""
import sys
from pathlib import Path

import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
res = np.load(src / "results.npy", allow_pickle=True).item()
n, reps = res["num_samples"], res["num_repetitions"]
for r in range(reps):
    for k in range(n):
        i = r * n + k
        joints = res["motion"][i].transpose(2, 0, 1)[: res["lengths"][i]]  # (T, 22, 3)
        np.save(out / f"p{k:02d}_r{r}.npy", joints)
print(f"[mdm] {n} prompts x {reps} reps -> {out}")
