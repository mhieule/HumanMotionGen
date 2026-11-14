"""Held-out denoising loss of one MDM checkpoint, per diffusion timestep.

Run from baselines/mdm with its venv:
    python ../../experiments/baseline/val_loss.py --build                      # once per machine
    python ../../experiments/baseline/val_loss.py --model_path save/RUN/model000005000.pt \
        --out save/RUN/eval/val_loss/s000005000.json

The loss is MDM's training loss (x0 MSE on valid frames) with the model in eval mode, on two
fixed sets: every val item and an equally large random subset of train (the gap between the two
shows overfitting). Items, captions, crops and noise are fixed by save/loss_sets.pt, so numbers
are comparable between checkpoints and between runs. Evaluated at 10 of the 50 timesteps
(TIMESTEPS), with the caption ("cond") and, on val, also without it ("uncond", the other half of
classifier-free guidance). Raw and EMA weights are both evaluated when the checkpoint has them.
One forward pass costs about 1 ms per motion, so one checkpoint takes about 1.5 min on a free GPU.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())  # baselines/mdm

ours = argparse.ArgumentParser(add_help=False)
ours.add_argument('--build', action='store_true', help='create the fixed loss sets and exit')
ours.add_argument('--sets', default='save/loss_sets.pt')
ours.add_argument('--out', default='')
ours.add_argument('--set_seed', type=int, default=1234)
mine, rest = ours.parse_known_args()
sys.argv = [sys.argv[0]] + rest

import numpy as np  # noqa: E402
import torch  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402

# PyTorch's memory-efficient attention kernel is wrong on the lab PC's GPU (see
# scripts/check_attention_kernel.py); the math kernel is what MDM's original PyTorch 1.x used.
torch.backends.cuda.enable_mem_efficient_sdp(False)

BATCH = 256
TIMESTEPS = list(range(2, 50, 5))  # 2, 7, ..., 47; 0 = almost clean, 49 = pure noise


def build():
    """One fixed draw (caption, crop) per item; val = all items, train = random subset of the same size."""
    from torch.utils.data import DataLoader, Subset
    from data_loaders.get_data import get_dataset
    from data_loaders.tensors import t2m_collate
    sets = {}
    n_val = None
    for split in ['val', 'train']:
        ds = get_dataset('humanml', num_frames=60, split=split, hml_mode='train')
        fixseed(mine.set_seed)
        if split == 'train':
            idx = np.sort(np.random.choice(len(ds), size=n_val, replace=False))
            ds = Subset(ds, idx.tolist())
        n_val = n_val or len(ds)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0,
                            collate_fn=lambda b: t2m_collate(b, len(b)))
        batches = []
        for motion, cond in loader:
            y = cond['y']
            batches.append({'motion': motion, 'mask': y['mask'], 'lengths': y['lengths'], 'text': y['text'],
                            'noise': torch.randn_like(motion)})
        sets[split] = batches
        print(f'{split}: {sum(len(b["text"]) for b in batches)} items in {len(batches)} batches')
    torch.save(sets, mine.sets)
    print('wrote', mine.sets)


@torch.no_grad()
def loss_per_t(model, diffusion, batches, dev, embeds, modes):
    out = {m: np.zeros(len(TIMESTEPS)) for m in modes}
    n = 0
    for b, emb in zip(batches, embeds):
        x, noise = b['motion'].to(dev), b['noise'].to(dev)
        y = {'mask': b['mask'].to(dev), 'lengths': b['lengths'].to(dev), 'text': b['text'], 'text_embed': emb}
        for i, t in enumerate(TIMESTEPS):
            ts = torch.full((x.shape[0],), t, device=dev, dtype=torch.long)
            for mode in modes:
                yy = dict(y, uncond=True) if mode == 'uncond' else y
                loss = diffusion.training_losses(model, x, ts, model_kwargs={'y': yy}, noise=noise)['loss']
                out[mode][i] += loss.sum().item()
        n += x.shape[0]
    return {k: (v / n).tolist() for k, v in out.items()}


def main():
    from utils.parser_util import evaluation_parser
    from utils import dist_util
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip
    args = evaluation_parser()
    dist_util.setup_dist(args.device)
    dev = dist_util.dev()
    sets = torch.load(mine.sets)
    model, diffusion = create_model_and_diffusion(args, SimpleNamespace(dataset=SimpleNamespace()))
    model.to(dev)
    model.eval()
    with torch.no_grad():  # the text encoder (CLIP) is frozen and not part of the checkpoint
        embeds = {s: [model.encode_text(b['text']) for b in batches] for s, batches in sets.items()}

    ckpt = torch.load(args.model_path, map_location='cpu')
    weights = {'raw': ckpt['model'], 'ema': ckpt['model_avg']} if 'model' in ckpt else {'raw': ckpt}
    step = int(Path(args.model_path).stem.replace('model', ''))
    result = {'step': step, 'model_path': args.model_path, 'sets': mine.sets, 'timesteps': TIMESTEPS,
              'n_items': {s: sum(len(b['text']) for b in v) for s, v in sets.items()}}
    for name, sd in weights.items():
        load_model_wo_clip(model, dict(sd))
        result[name] = {}
        for split, batches in sets.items():
            modes = ['cond', 'uncond'] if split == 'val' else ['cond']
            per_t = loss_per_t(model, diffusion, batches, dev, embeds[split], modes)
            for mode, v in per_t.items():
                result[name][f'{split}_{mode}'] = v
        r = result[name]
        print(f'step {step} {name}: val {np.mean(r["val_cond"]):.5f}  train {np.mean(r["train_cond"]):.5f}  '
              f'val uncond {np.mean(r["val_uncond"]):.5f}')
    out = Path(mine.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result))
    os.replace(tmp, out)


if __name__ == '__main__':
    build() if mine.build else main()
