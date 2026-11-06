"""SALAD: prompts -> joint positions, one .npy per (prompt, repetition).

Run from baselines/salad with its .venv (compare.sh does this).
Mirrors the repo's playground.ipynb: DDIM, 50 steps, cfg 7.5.
"""
import argparse
import os
import sys
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from t2m import Text2Motion  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402
from utils.get_opt import get_opt  # noqa: E402
from utils.motion_process import recover_from_ric  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--out", required=True)
p.add_argument("--frames", type=int, default=120)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--seed", type=int, default=10)
p.add_argument("--cfg_scale", type=float, default=7.5)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--denoiser", default="t2m_denoiser_vpred_vaegelu")
a = p.parse_args()

assert a.frames % 4 == 0, "frames must be divisible by 4"
prompts = [l.strip() for l in open(a.prompts) if l.strip()]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

gen = Text2Motion(a.denoiser, "t2m")
wrapper_opt = get_opt(gen.opt.dataset_opt_path, gen.device)
mean = np.load(pjoin(wrapper_opt.meta_dir, "mean.npy"))
std = np.load(pjoin(wrapper_opt.meta_dir, "std.npy"))

for r in range(a.reps):
    for k, text in enumerate(prompts):
        fixseed(a.seed + r)
        _, motion, _ = gen.generate(text, a.frames, a.cfg_scale, a.steps)
        feats = motion[0].cpu().numpy() * std + mean  # (T, 263)
        joints = recover_from_ric(torch.from_numpy(feats).float(), 22).numpy()
        np.save(out / f"p{k:02d}_r{r}.npy", joints)
    print(f"[salad] rep {r}: {len(prompts)} prompts done")
