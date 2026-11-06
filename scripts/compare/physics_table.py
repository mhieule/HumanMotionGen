"""Physical-plausibility table for a comparison run, with real HumanML3D clips as reference.

    python scripts/compare/physics_table.py comparisons/<run> [--gt-max 300]

Uses stats() from scripts/motion_stats.py on every <run>/<model>/*.npy and on the first
--gt-max non-mirrored test clips. Prints model means and writes <run>/physics.md.
Small samples (prompts x reps): read it as a sanity check, not a benchmark.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts"))
from motion_stats import stats  # noqa: E402

KEYS = [("float_pct", "floating %", "{:.0f}"), ("sink_cm", "sinking cm", "{:.1f}"),
        ("slide_cms", "foot slide cm/s", "{:.1f}"), ("slide_pct", "slide > 10 cm/s %", "{:.0f}"),
        ("jerk", "jerk m/s³", "{:.0f}")]
NAMES = {"gt": "Real (HumanML3D test)", "mdm": "MDM", "salad": "SALAD", "mhf": "MotionHiFlow"}

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--gt-max", type=int, default=300)
a = p.parse_args()
run = Path(a.run)

rows = {}
for m in ["mdm", "salad", "mhf"]:
    files = sorted((run / m).glob("p*_r*.npy"))
    if files:
        rows[m] = [stats(np.load(f))[0] for f in files]

data = root / "mdm_assets/dataset/HumanML3D"
ids = [i for i in (data / "test.txt").read_text().split() if not i.startswith("M")][: a.gt_max]
rows["gt"] = [stats(np.load(data / "new_joints" / f"{i}.npy"))[0] for i in ids]

head = "| Model | n | " + " | ".join(h for _, h, _ in KEYS) + " |"
lines = [head, "|" + "---|" * (len(KEYS) + 2)]
for m in ["gt", "mdm", "salad", "mhf"]:
    if m not in rows:
        continue
    vals = [fmt.format(np.nanmean([s[k] for s in rows[m]])) for k, _, fmt in KEYS]
    lines.append(f"| {NAMES[m]} | {len(rows[m])} | " + " | ".join(vals) + " |")
table = "\n".join(lines)
print(table)
(run / "physics.md").write_text(
    f"Prompts: {run / 'prompts.txt'}. Lower is better for all columns; the real row is the target.\n\n"
    + table + "\n")
