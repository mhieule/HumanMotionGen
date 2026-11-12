"""Sanity checks for the Tier 0 code. Run from baselines/mdm:
    python ../../experiments/tier0/checks.py --model_path save/humanml_enc_512_50steps/model000750000.pt
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch  # noqa: E402
from utils.parser_util import evaluation_parser  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402
from utils import dist_util  # noqa: E402
from utils.model_util import create_model_and_diffusion, load_saved_model  # noqa: E402
from utils.sampler_util import ClassifierFreeSampleModel  # noqa: E402
from data_loaders.get_data import get_dataset_loader  # noqa: E402
from guidance import GuidedModel  # noqa: E402
from scoring import TMRScorer  # noqa: E402

args = evaluation_parser()
dist_util.setup_dist(args.device)
dev = dist_util.dev()
fixseed(0)
loader = get_dataset_loader(name='humanml', batch_size=32, num_frames=None, split='val', hml_mode='eval',
                            fixed_len=0, pred_len=0, device=dev, autoregressive=False)
model, diffusion = create_model_and_diffusion(args, loader)
load_saved_model(model, args.model_path, use_avg=args.use_ema)
model.to(dev)
model.eval()

motion, cond = next(iter(loader))
y = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in cond['y'].items()}
y['scale'] = torch.full((32,), 2.5, device=dev)
x = torch.randn_like(motion.to(dev))
t = torch.full((32,), 30, device=dev, dtype=torch.long)

valid = y['mask'].expand_as(x)  # compare on real (unpadded) frames only
with torch.no_grad():
    a, b = model(x, t, dict(y)), model(x, t, dict(y))
    ref = ClassifierFreeSampleModel(model)(x, t, dict(y))
    ours = GuidedModel(model, scale=2.5)(x, t, dict(y))
    apg1 = GuidedModel(model, scale=2.5, apg=dict(eta=1.0))(x, t, dict(y))
    apg0 = GuidedModel(model, scale=2.5, apg=dict(eta=0.0))(x, t, dict(y))
mx = lambda d: d.abs()[valid].max().item()
print(f'[0] MDM run-to-run noise (same call twice, GPU kernels not deterministic): max |diff| = {mx(a - b):.1e}')
print(f'[1] CFG wrapper vs MDM ClassifierFreeSampleModel: max |diff| = {mx(ref - ours):.1e} (should be ~[0])')
print(f'[2] APG(eta=1) vs CFG: max |diff| = {mx(apg1 - ours):.1e} (should be ~[0]); '
      f'APG(eta=0) vs CFG: mean |diff| = {(apg0 - ours).abs()[valid].mean():.3f}')

tmr = TMRScorer(dev)
ds = loader.dataset.t2m_dataset
mean, std = (torch.tensor(a, dtype=torch.float32, device=dev) for a in (ds.mean, ds.std))
hits, n, matched, other = 0, 0, [], []
for i, (motion, cond) in enumerate(loader):
    raw = motion.to(dev).squeeze(2).permute(0, 2, 1) * std + mean
    mz = tmr.encode_motion(raw, cond['y']['lengths'])
    tz = tmr.encode_text(cond['y']['text'])
    sim = tz @ mz.T
    hits += (sim.argmax(1) == torch.arange(32, device=dev)).sum().item()
    n += 32
    matched.append(sim.diag().mean().item())
    other.append(((sim.sum() - sim.diag().sum()) / (32 * 31)).item())
    if i == 15:
        break
print(f'[3] TMR on real val motions: text->motion R@1 within batches of 32 = {hits / n:.3f} (chance 0.031); '
      f'cos sim matched {np.mean(matched):.3f} vs. mismatched {np.mean(other):.3f}')
