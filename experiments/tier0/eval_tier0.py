"""Evaluate one Tier 0 sampling configuration of a trained MDM on HumanML3D.

Run from baselines/mdm with its venv (run_tier0.sh does this):
    python ../../experiments/tier0/eval_tier0.py --model_path save/humanml_enc_512_50steps/model000750000.pt \
        --name cfg2.5 --split val --reps 5 --scale 2.5 [--interval 8 43] [--apg_eta 0 --apg_momentum -0.5]
        [--best_of 4 --verifier tmr]

Metrics per replication: MDM's own FID / R-precision / matching score / diversity (Guo et al.
evaluator, 1000 samples), plus TMR text-motion similarity, physics stats, sampling time and
network passes (NFE) per generated motion. Replication k reseeds with seed+k, so every config
sees the same prompts and (for 1-candidate configs) the same initial noise in replication k.
Writes <out_dir>/<name>.json and <out_dir>/<name>.log.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.getcwd())  # baselines/mdm

ours = argparse.ArgumentParser(add_help=False)
ours.add_argument('--name', required=True)
ours.add_argument('--out_dir', default='save/tier0')
ours.add_argument('--split', default='val', choices=['val', 'test'])
ours.add_argument('--reps', type=int, default=5)
ours.add_argument('--num_samples', type=int, default=1000)
ours.add_argument('--scale', type=float, default=2.5)
ours.add_argument('--interval', type=int, nargs=2, default=None, metavar=('T_LO', 'T_HI'))
ours.add_argument('--apg_eta', type=float, default=None, help='enables APG')
ours.add_argument('--apg_momentum', type=float, default=0.0)
ours.add_argument('--apg_norm', type=float, default=0.0, help='RMS threshold of the guidance update')
ours.add_argument('--best_of', type=int, default=1)
ours.add_argument('--verifier', default='tmr', choices=['tmr', 'tmr_phys'])
ours.add_argument('--gt_reference', action='store_true', help='also score real motions (TMR, physics)')
mine, rest = ours.parse_known_args()
sys.argv = [sys.argv[0]] + rest

import numpy as np  # noqa: E402
import torch  # noqa: E402
from utils.parser_util import evaluation_parser  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402
from utils import dist_util  # noqa: E402
from utils.model_util import create_model_and_diffusion, load_saved_model  # noqa: E402
from data_loaders.get_data import get_dataset_loader  # noqa: E402
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader  # noqa: E402
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper  # noqa: E402
from eval.eval_humanml import evaluate_matching_score, evaluate_fid, evaluate_diversity  # noqa: E402
from guidance import GuidedModel, BestOfNDiffusion  # noqa: E402
from scoring import TMRScorer, make_verifier, physics, PHYS_KEYS  # noqa: E402

torch.multiprocessing.set_sharing_strategy('file_system')


def summarize(values):
    a = np.array(values, dtype=np.float64)
    return {'mean': a.mean(0).tolist(), 'ci95': (1.96 * a.std(0) / np.sqrt(len(a))).tolist(),
            'values': a.tolist()}


def extra_metrics(tmr, raw, lengths, texts, bs=256):
    sims, phys = [], {k: [] for k in PHYS_KEYS}
    for i in range(0, len(raw), bs):
        r, l, t = raw[i:i + bs], lengths[i:i + bs], texts[i:i + bs]
        sims.append(tmr.similarity(r, l, t).cpu().numpy())
        p = physics(r, l)
        for k in PHYS_KEYS:
            phys[k].append(p[k])
    out = {'tmr_sim': float(np.concatenate(sims).mean())}
    out.update({k: float(np.nanmean(np.concatenate(v))) for k, v in phys.items()})
    return out


def main():
    args = evaluation_parser()
    args.batch_size = 32  # MDM: must be 32, R-precision is computed within batches of 32
    out_dir = Path(mine.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'{mine.name}.log'
    dist_util.setup_dist(args.device)
    dev = dist_util.dev()
    fixseed(args.seed)

    gt_loader = get_dataset_loader(name=args.dataset, batch_size=32, num_frames=None, split=mine.split, hml_mode='gt')
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=32, num_frames=None, split=mine.split, hml_mode='eval',
                                    fixed_len=0, pred_len=0, device=dev, autoregressive=False)
    model, diffusion = create_model_and_diffusion(args, gen_loader)
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    model.to(dev)
    model.eval()

    apg = None if mine.apg_eta is None else dict(eta=mine.apg_eta, momentum=mine.apg_momentum,
                                                  norm_threshold=mine.apg_norm)
    gm = GuidedModel(model, scale=mine.scale, interval=mine.interval, apg=apg,
                     num_steps=args.diffusion_steps).to(dev).eval()

    ds = gen_loader.dataset.t2m_dataset
    mean = torch.tensor(ds.mean, dtype=torch.float32, device=dev)
    std = torch.tensor(ds.std, dtype=torch.float32, device=dev)
    tmr = TMRScorer(dev)
    sampler = diffusion
    if mine.best_of > 1:
        sampler = BestOfNDiffusion(diffusion, mine.best_of, make_verifier(mine.verifier, tmr, mean, std))

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dev)
    per_rep = {k: [] for k in ['matching', 'r_precision', 'fid', 'diversity', 'gt_matching', 'gt_r_precision', 'gt_diversity',
                               'sample_sec', 'nfe_per_motion', 'tmr_sim'] + PHYS_KEYS}
    result = {'name': mine.name, 'config': vars(mine), 'model_path': args.model_path, 'seed': args.seed,
              'gpu': torch.cuda.get_device_name(0), 'diffusion_steps': args.diffusion_steps}

    with open(log_path, 'w') as f:
        print(json.dumps(result['config']), file=f, flush=True)
        if mine.gt_reference:
            fixseed(args.seed)
            raw, lens, texts = [], [], []
            for motion, cond in gen_loader:
                raw.append((motion.to(dev).squeeze(2).permute(0, 2, 1) * std + mean))
                lens.append(cond['y']['lengths'])
                texts += cond['y']['text']
                if len(texts) >= mine.num_samples:
                    break
            result['gt_extras'] = extra_metrics(tmr, torch.cat(raw), torch.cat(lens).to(dev), texts)
            print('GT extras', result['gt_extras'], file=f, flush=True)

        for rep in range(mine.reps):
            fixseed(args.seed + rep)
            gm.nfe, traj_calls = 0, [0]
            orig_forward = gm.forward

            def counting_forward(x, timesteps, y=None):
                if int(timesteps[0].item()) == args.diffusion_steps - 1:
                    traj_calls[0] += 1
                return orig_forward(x, timesteps, y)
            gm.forward = counting_forward

            torch.cuda.synchronize()
            t0 = time.time()
            motion_loader, _ = get_mdm_loader(args, model=gm, diffusion=sampler, batch_size=32,
                                              ground_truth_loader=gen_loader, mm_num_samples=0, mm_num_repeats=0,
                                              max_motion_length=gt_loader.dataset.opt.max_motion_length,
                                              num_samples_limit=mine.num_samples, scale=mine.scale)
            torch.cuda.synchronize()
            sample_sec = time.time() - t0
            gm.forward = orig_forward
            n_motions = len(motion_loader.dataset)
            nfe_per_motion = gm.nfe / max(traj_calls[0], 1) * mine.best_of

            print(f'== rep {rep}: {n_motions} motions, {sample_sec:.0f}s sampling, NFE/motion {nfe_per_motion:.0f}',
                  file=f, flush=True)
            loaders = {'ground truth': gt_loader, 'vald': motion_loader}
            mat, rprec, acti = evaluate_matching_score(eval_wrapper, loaders, f)
            fid = evaluate_fid(eval_wrapper, gt_loader, acti, f)
            div = evaluate_diversity(acti, f, min(300, n_motions - 1))  # 300 = MDM default (needs > 300 motions)

            gens = motion_loader.dataset.generated_motion
            raw = torch.tensor(np.stack([g['motion'] for g in gens]), device=dev) * std + mean
            lens = torch.tensor([int(g['length']) for g in gens], device=dev)
            ex = extra_metrics(tmr, raw, lens, [g['caption'] for g in gens])
            print('extras', ex, file=f, flush=True)

            per_rep['matching'].append(mat['vald']); per_rep['r_precision'].append(rprec['vald'])
            per_rep['fid'].append(fid['vald']); per_rep['diversity'].append(div['vald'])
            per_rep['gt_matching'].append(mat['ground truth']); per_rep['gt_diversity'].append(div['ground truth']); per_rep['gt_r_precision'].append(rprec['ground truth'])
            per_rep['sample_sec'].append(sample_sec); per_rep['nfe_per_motion'].append(nfe_per_motion)
            for k, v in ex.items():
                per_rep[k].append(v)

            result['reps_done'] = rep + 1
            result['metrics'] = {k: summarize(v) for k, v in per_rep.items() if v}
            tmp = out_dir / f'{mine.name}.json.tmp'
            tmp.write_text(json.dumps(result, indent=1))
            os.replace(tmp, out_dir / f'{mine.name}.json')

    result['complete'] = True
    (out_dir / f'{mine.name}.json').write_text(json.dumps(result, indent=1))
    m = result['metrics']
    print(f"{mine.name}: FID {m['fid']['mean']:.3f}±{m['fid']['ci95']:.3f}  R@3 {m['r_precision']['mean'][2]:.3f}  "
          f"MM {m['matching']['mean']:.3f}  Div {m['diversity']['mean']:.3f}  TMR {m['tmr_sim']['mean']:.3f}  "
          f"{np.mean(per_rep['sample_sec']):.0f}s/rep  NFE {np.mean(per_rep['nfe_per_motion']):.0f}")


if __name__ == '__main__':
    main()
