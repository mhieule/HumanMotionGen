"""MotionHiFlow: prompts -> joint positions, one .npy per (prompt, repetition).

Run from baselines/MotionHiFlow with its .venv (compare.sh does this).
Same core as the repo's gen_t2m.py (preset `tmdit`: cond_scale 4.5, 12 steps),
minus its BVH/IK export and rendering, which are slow and not needed here.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from src.utils import load_model, seed_everything  # noqa: E402
from src.utils.motion_process import recover_from_ric  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--out", required=True)
p.add_argument("--frames", type=int, default=120)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--seed", type=int, default=10)
p.add_argument("--cond_scale", type=float, default=4.5)
p.add_argument("--time_steps", type=int, default=12)
a = p.parse_args()

assert a.frames % 4 == 0, "frames must be divisible by 4"
prompts = [l.strip() for l in open(a.prompts) if l.strip()]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
device = "cuda"

flow = load_model("logs/t2m_tmdit_16d").to(device).eval()
vae = load_model("logs/t2m_vae_agcn_16d").to(device).eval()
meta = "deps/evaluators/t2m/Comp_v6_KLD005/meta"  # repo default for t2m (resources.yaml)
mean, std = np.load(f"{meta}/mean.npy"), np.load(f"{meta}/std.npy")

for r in range(a.reps):
    seed_everything(a.seed + r)
    m_tok = torch.full((len(prompts),), a.frames // 4, dtype=torch.long, device=device)
    with torch.no_grad():
        lat, pred_tok = flow.generate(text=prompts, m_lengths=m_tok, time_steps=a.time_steps,
                                      cond_scale=a.cond_scale, cfg_interval=[0.0, 1.0])
        feats = vae.decode(lat).cpu().numpy() * std + mean  # (N, T, 263)
    for k, f in enumerate(feats):
        n = int(pred_tok[k]) * 4
        joints = recover_from_ric(torch.from_numpy(f[:n]).float(), 22).numpy()
        np.save(out / f"p{k:02d}_r{r}.npy", joints)
    print(f"[mhf] rep {r}: {len(prompts)} prompts done")
