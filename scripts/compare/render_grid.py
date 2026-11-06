"""Render one grid video per prompt: rows = models, columns = repetitions.

Reads <run>/<model>/pKK_rR.npy (T, 22, 3) for every model folder present and
writes <run>/grid_pKK.mp4. Each panel has its own camera centred on that motion;
the root's ground trajectory is drawn in grey.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
         [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
COLORS = ["#d62728", "#1f77b4", "#222222", "#ff7f0e", "#2ca02c"]  # r-leg, l-leg, spine, r-arm, l-arm
ORDER = ["gt", "mdm", "salad", "mhf"]
LABEL = {"gt": "real (HumanML3D)", "mdm": "MDM (diffusion, 2022)",
         "salad": "SALAD (diffusion, 2025)", "mhf": "MotionHiFlow (flow matching, 2026)"}

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--fps", type=int, default=20)
a = p.parse_args()
run = Path(a.run)
prompts = [l.strip() for l in open(run / "prompts.txt") if l.strip()]
models = [m for m in ORDER if (run / m).is_dir()]
reps = len(list((run / models[0]).glob("p00_r*.npy")))

for k, text in enumerate(prompts):
    data = {(m, r): np.load(run / m / f"p{k:02d}_r{r}.npy") for m in models for r in range(reps)
            if (run / m / f"p{k:02d}_r{r}.npy").exists()}
    T = max(len(j) for j in data.values())
    fig = plt.figure(figsize=(3.6 * reps, 3.6 * len(models) + 0.5))
    fig.suptitle(text, fontsize=12)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - 0.5 / fig.get_figheight(), wspace=0, hspace=0)
    panels = []
    for i, m in enumerate(models):
        for r in range(reps):
            ax = fig.add_subplot(len(models), reps, i * reps + r + 1, projection="3d")
            ax.set_axis_off()
            ax.view_init(elev=20, azim=-60)
            ax.text2D(0.03, 0.93, LABEL[m] if r == 0 else f"seed +{r}", transform=ax.transAxes,
                      fontsize=10, weight="bold" if r == 0 else "normal")
            if (m, r) not in data:
                continue
            j = data[(m, r)].copy()
            j[..., 1] -= j[..., 1].min()  # feet on the floor
            c = (j[:, 0, [0, 2]].max(0) + j[:, 0, [0, 2]].min(0)) / 2
            rad = max(1.0, np.abs(j[:, :, [0, 2]] - c).max() + 0.2)
            # matplotlib 3D: plot (x, z, y) so y is up; equal scale on all axes
            ax.set_xlim(c[0] - rad, c[0] + rad)
            ax.set_ylim(c[1] - rad, c[1] + rad)
            ax.set_zlim(0, 2 * rad)
            ax.set_box_aspect((1, 1, 1), zoom=1.35)
            xx, zz = np.meshgrid([c[0] - rad, c[0] + rad], [c[1] - rad, c[1] + rad])
            ax.plot_surface(xx, zz, 0 * xx, color="#e8e8e8", alpha=0.6, shade=False)
            ax.plot(j[:, 0, 0], j[:, 0, 2], 0 * j[:, 0, 1], color="#999999", lw=1)
            lines = [ax.plot([], [], [], color=col, lw=2)[0] for col in COLORS]
            panels.append((j, lines))

    writer = FFMpegWriter(fps=a.fps)
    path = run / f"grid_p{k:02d}.mp4"
    with writer.saving(fig, str(path), dpi=100):
        for t in range(T):
            for j, lines in panels:
                f = j[min(t, len(j) - 1)]
                for chain, line in zip(CHAIN, lines):
                    line.set_data(f[chain, 0], f[chain, 2])
                    line.set_3d_properties(f[chain, 1])
            writer.grab_frame()
    plt.close(fig)
    print(f"[render] {path}")
