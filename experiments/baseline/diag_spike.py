"""Find what causes a gradient spike in MDM training.

Run from baselines/mdm with its venv:
  capture:  python ../../experiments/baseline/diag_spike.py capture <normal train args, --num_steps N>
            Trains exactly like train_instrumented.py; whenever the gradient norm of a step exceeds
            SPIKE, it writes <save_dir>/spike_<step>.pt (weights before the update, the batch, and
            the random state the step started from) and stops after MAX_DUMPS dumps.
  analyze:  python ../../experiments/baseline/diag_spike.py analyze <save_dir>/spike_<step>.pt
            Replays the step from the dump, checks the gradient norm reproduces, then splits it
            by sample and by parameter group, and repeats the replay with --mask_frames semantics
            and without dropout.
"""
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
mode = sys.argv.pop(1)

import numpy as np  # noqa: E402
import torch  # noqa: E402

SPIKE, MAX_DUMPS = 30.0, 3


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(), 'np': np.random.get_state(),
            'py': random.getstate()}


def set_rng_state(s):
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state(s['cuda'])
    np.random.set_state(s['np']); random.setstate(s['py'])


def grad_norm(params):
    return float(sum((p.grad.float() ** 2).sum().item() for p in params if p.grad is not None) ** 0.5)


def capture():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_instrumented as ti

    class CaptureLoop(ti.InstrumentedTrainLoop):
        dumps = 0

        def run_step(self, batch, cond):
            if self.total_step() > self.num_steps:
                raise ti.StopTraining
            state = rng_state()
            self.forward_backward(batch, cond)          # gradients at the weights before this step's update
            gn = grad_norm(self.mp_trainer.master_params)
            if gn > SPIKE:
                sd = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items() if not k.startswith('clip_model.')}
                y = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in cond['y'].items()}
                path = Path(self.save_dir) / f'spike_{self.total_step():06d}.pt'
                torch.save({'step': self.total_step(), 'grad_norm': gn, 'model': sd, 'batch': batch.cpu(), 'y': y,
                            'rng': state, 'args': vars(self.args)}, path)
                print(f'SPIKE at step {self.total_step()}: grad norm {gn:.1f} -> {path}', flush=True)
                CaptureLoop.dumps += 1
            # the rest of MDM's run_step, unchanged
            self.mp_trainer.optimize(self.opt)
            self.update_average_model()
            self._anneal_lr()
            self.log_step()
            if CaptureLoop.dumps >= MAX_DUMPS:
                raise ti.StopTraining

    ti.stock_loop.log_loss_dict = ti.log_loss_dict
    ti.stock.TrainLoop = CaptureLoop
    try:
        ti.stock.main()
    except ti.StopTraining:
        print('capture finished')


def analyze():
    from utils import dist_util
    from utils.model_util import create_model_and_diffusion
    from diffusion.resample import create_named_schedule_sampler
    d = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    args = SimpleNamespace(**d['args'])
    dist_util.setup_dist(0)
    dev = dist_util.dev()
    model, diffusion = create_model_and_diffusion(args, SimpleNamespace(dataset=SimpleNamespace()))
    missing, unexpected = model.load_state_dict(d['model'], strict=False)
    assert not unexpected and all(k.startswith('clip_model.') for k in missing)
    model.to(dev)
    sampler = create_named_schedule_sampler('uniform', diffusion)
    batch = d['batch'].to(dev)
    y = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d['y'].items()}
    params = [p for n, p in model.named_parameters() if not n.startswith('clip_model.')]
    names = [n for n, p in model.named_parameters() if not n.startswith('clip_model.')]

    def replay(train=True, mask_frames=False, per_sample=False):
        """Recompute the step's loss from the saved random state. Returns t, per-sample loss, extras."""
        model.train(train)
        model.mask_frames = mask_frames
        set_rng_state(d['rng'])
        t, w = sampler.sample(batch.shape[0], dev)
        seen = {}
        hook = model.embed_text.register_forward_hook(lambda m, i, o: seen.__setitem__('cond_in', i[0].detach()))
        losses = diffusion.training_losses(model, batch, t, model_kwargs={'y': dict(y)})['loss'] * w
        hook.remove()
        dropped = (seen['cond_in'].abs().sum(-1) == 0).flatten().cpu().numpy()   # caption dropped (CFG training)
        model.zero_grad()
        if not per_sample:
            losses.mean().backward()
            return t, losses.detach(), dropped, grad_norm(params)
        per = []
        for i in range(batch.shape[0]):
            model.zero_grad()
            (losses[i] / batch.shape[0]).backward(retain_graph=True)
            per.append(grad_norm(params))
        return t, losses.detach(), dropped, np.array(per)

    print(f"dump: step {d['step']}, gradient norm during training {d['grad_norm']:.1f}")
    t, loss, dropped, gn = replay()
    print(f'replay (train mode, as trained):        grad norm {gn:8.2f}   loss {loss.mean().item():.4f}')
    by_group = {}
    for n, p in zip(names, params):
        g = '.'.join(n.split('.')[:3]) if n.startswith('seqTransEncoder') else n.split('.')[0]
        by_group[g] = by_group.get(g, 0.0) + (p.grad ** 2).sum().item()
    print('  by parameter group (norm):', {k: round(v ** 0.5, 2) for k, v in sorted(by_group.items(), key=lambda kv: -kv[1])[:8]})
    for label, kw in [('replay, dropout off (eval mode)', dict(train=False)),
                      ('replay, train mode + mask_frames', dict(mask_frames=True)),
                      ('replay, eval mode + mask_frames', dict(train=False, mask_frames=True))]:
        _, l2, _, g2 = replay(**kw)
        print(f'{label:39s} grad norm {g2:8.2f}   loss {l2.mean().item():.4f}')

    t, loss, dropped, per = replay(per_sample=True)
    lengths = y['lengths'].cpu().numpy()
    order = np.argsort(-per)
    print(f'\nper-sample gradient norm: median {np.median(per):.3f}, max {per.max():.2f}; top samples:')
    for i in order[:6]:
        print(f'  #{i:2d} grad {per[i]:8.2f}  loss {loss[i].item():.3f}  t={t[i].item():2d}  length={lengths[i]:3d} of {batch.shape[-1]} '
              f'(padding {batch.shape[-1] - lengths[i]:3d})  caption dropped={bool(dropped[i])}  "{y["text"][i][:70]}"')
    print('batch: lengths min/median/max', lengths.min(), int(np.median(lengths)), lengths.max(),
          '| captions dropped:', int(dropped.sum()), '| input abs max', batch.abs().max().item())

    # Same weights, batch, timesteps, noise and text embedding on GPU (float32) and CPU (float32, float64).
    # Dropout / caption-drop masks cannot be shared between devices, so compare eval mode exactly and
    # train mode as a distribution over 8 dropout seeds.
    import copy
    set_rng_state(d['rng'])
    t, _ = sampler.sample(batch.shape[0], dev)
    noise = torch.randn_like(batch)
    with torch.no_grad():
        emb = model.encode_text(y['text'])
    model.mask_frames = False

    def fixed(m, dev_, dtype, train, seed=0):
        m.train(train)
        ps = [p for n, p in m.named_parameters() if not n.startswith('clip_model.')]
        yy = {'mask': y['mask'].to(dev_), 'lengths': y['lengths'].to(dev_), 'text': y['text'],
              'text_embed': emb.to(dev_, dtype)}
        torch.manual_seed(seed)
        losses = diffusion.training_losses(m, batch.to(dev_, dtype), t.to(dev_), model_kwargs={'y': yy},
                                           noise=noise.to(dev_, dtype))['loss']
        m.zero_grad()
        losses.mean().backward()
        return grad_norm(ps), losses.mean().item()

    cpu = copy.deepcopy(model)
    cpu.cpu()          # MDM's _apply returns None, so no chaining
    rows = [('GPU float32', model, dev, torch.float32), ('CPU float32', cpu, 'cpu', torch.float32)]
    print('\nfixed timesteps / noise / text embedding:')
    res = {}
    for label, m, dev_, dtype in rows + [('CPU float64', None, 'cpu', torch.float64)]:
        if m is None:
            cpu.double()
            m = cpu
        g, l = fixed(m, dev_, dtype, train=False)
        gs = [fixed(m, dev_, dtype, train=True, seed=k)[0] for k in range(8)]
        res[label] = {'eval': g, 'train_seeds': gs}
        print(f'  {label}: eval mode grad norm {g:10.3f} (loss {l:.4f}) | train mode, 8 dropout seeds: '
              f'median {np.median(gs):.2f}, max {max(gs):.2f}  {[round(x, 1) for x in gs]}')
    out = Path(sys.argv[1]).with_suffix('.json')
    out.write_text(json.dumps({'step': d['step'], 'per_sample_grad': per.tolist(), 't': t.cpu().tolist(),
                               'lengths': lengths.tolist(), 'dropped': dropped.tolist(),
                               'loss': loss.cpu().tolist(), 'devices': res}))


if __name__ == '__main__':
    capture() if mode == 'capture' else analyze()
