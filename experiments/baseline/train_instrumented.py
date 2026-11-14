"""Instrumented launcher for MDM training. The training itself is MDM's, unchanged.

Run from baselines/mdm with its venv, with exactly the arguments of `python -m train.train_mdm`:
    python ../../experiments/baseline/train_instrumented.py --save_dir save/E003_mdm50_baseline --overwrite \
        --dataset humanml --diffusion_steps 50 --num_steps 750000 --save_interval 5000 \
        --log_interval 100 --use_ema --train_platform_type TensorboardPlatform

It calls MDM's own `train.train_mdm.main()` and only swaps `TrainLoop` for a subclass that
  - writes <save_dir>/train_log.jsonl and extra TensorBoard scalars every --log_interval steps
    (loss per diffusion timestep, gradient / weight norm, raw-vs-EMA distance, speed, GPU memory),
  - stops at exactly --num_steps, also after a resume (MDM's loop counts epochs and, after a
    resume, runs the full epoch count again),
  - writes <save_dir>/run_info.json (commits, versions, arguments).
Nothing here touches the model, the optimizer, the data order or any random stream;
check_equivalence.sh verifies that against stock MDM.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())  # baselines/mdm

import numpy as np  # noqa: E402
import torch  # noqa: E402

# PyTorch's memory-efficient attention kernel is wrong on the lab PC's GPU (see
# scripts/check_attention_kernel.py); the math kernel is what MDM's original PyTorch 1.x used.
torch.backends.cuda.enable_mem_efficient_sdp(False)

import train.train_mdm as stock  # noqa: E402
import train.training_loop as stock_loop  # noqa: E402
from diffusion import logger  # noqa: E402

HERE = Path(__file__).resolve().parent
EVAL_MARKER = 'eval/RUNNING'  # written by watch.py while an evaluation shares the GPU


class StopTraining(Exception):
    pass


class LossByTimestep:
    """Mean training loss per diffusion timestep over one log interval."""

    def __init__(self, num_timesteps):
        self.sum = np.zeros(num_timesteps)
        self.cnt = np.zeros(num_timesteps, dtype=np.int64)

    def add(self, ts, losses):
        np.add.at(self.sum, ts, losses)
        np.add.at(self.cnt, ts, 1)

    def pop(self):
        mean = [round(float(s / c), 6) if c else None for s, c in zip(self.sum, self.cnt)]
        self.sum[:] = 0
        self.cnt[:] = 0
        return mean


_by_t = None
_stock_log_loss_dict = stock_loop.log_loss_dict


def log_loss_dict(diffusion, ts, losses):
    _stock_log_loss_dict(diffusion, ts, losses)
    _by_t.add(ts.cpu().numpy(), losses['loss'].detach().cpu().numpy())


def git(*args, cwd):
    try:
        return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ''


class InstrumentedTrainLoop(stock_loop.TrainLoop):
    def __init__(self, args, train_platform, model, diffusion, data):
        global _by_t
        super().__init__(args, train_platform, model, diffusion, data)
        assert self.num_steps % self.save_interval == 0, 'num_steps must be a multiple of save_interval'
        _by_t = LossByTimestep(diffusion.num_timesteps)
        self.steps_per_epoch = len(data)
        self.log_path = Path(self.save_dir) / 'train_log.jsonl'
        self.train_sec_before = 0.0  # training time of earlier sessions (resume)
        if self.resume_step and self.log_path.exists():
            # drop log lines past the checkpoint we resume from, keep the accumulated time
            kept = [l for l in self.log_path.read_text().splitlines() if l and json.loads(l)['step'] <= self.resume_step]
            self.log_path.write_text(''.join(l + '\n' for l in kept))
            if kept:
                self.train_sec_before = json.loads(kept[-1])['train_sec']
        elif self.log_path.exists():
            self.log_path.unlink()
        self.session_start = time.time()
        self.t_prev_end = None
        self.acc = {'step_sec': 0.0, 'data_sec': 0.0, 'n': 0}
        self.t_interval = time.time()
        self.write_run_info()

    def write_run_info(self):
        repo = HERE.parents[1]
        info = {
            'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'resume_step': self.resume_step,
            'argv': sys.argv[1:], 'args': vars(self.args),
            'repo_commit': git('rev-parse', 'HEAD', cwd=repo), 'repo_dirty': git('status', '--porcelain', cwd=repo),
            'mdm_commit': git('rev-parse', 'HEAD', cwd=os.getcwd()),
            'mdm_dirty': git('status', '--porcelain', '--untracked-files=no', cwd=os.getcwd()),
            'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'mem_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'steps_per_epoch': self.steps_per_epoch, 'train_items': len(self.data.dataset),
            'params_wo_clip': sum(p.numel() for p in self.model.parameters_wo_clip()),
        }
        path = Path(self.save_dir) / 'run_info.json'
        runs = json.loads(path.read_text()) if path.exists() and self.resume_step else []
        path.write_text(json.dumps(runs + [info], indent=1))

    def run_step(self, batch, cond):
        if self.total_step() > self.num_steps:
            self.train_platform.close()
            raise StopTraining
        t0 = time.perf_counter()
        if self.t_prev_end is not None:
            self.acc['data_sec'] += t0 - self.t_prev_end
        super().run_step(batch, cond)
        self.t_prev_end = time.perf_counter()
        self.acc['step_sec'] += self.t_prev_end - t0
        self.acc['n'] += 1
        if self.total_step() % self.log_interval == 0:
            self.write_log()

    @torch.no_grad()
    def ema_rel_dist(self):
        num = den = 0.0
        for p, a in zip(self.mp_trainer.master_params, self.model_avg.parameters()):
            num += (p - a).pow(2).sum().item()
            den += p.pow(2).sum().item()
        return (num / den) ** 0.5

    def write_log(self):
        # MDM's loop dumps (and resets) the same logger values right after this returns.
        kv = dict(logger.get_current().name2val)
        now, n = time.time(), max(self.acc['n'], 1)
        step = self.total_step()
        rec = {
            'step': step, 'epoch': round(step / self.steps_per_epoch, 3),
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'train_sec': round(self.train_sec_before + now - self.session_start, 1),
            'steps_per_sec': round(n / (now - self.t_interval), 3),
            'step_ms': round(1e3 * self.acc['step_sec'] / n, 2),
            'data_ms': round(1e3 * self.acc['data_sec'] / n, 2),
            'eval_running': (Path(self.save_dir) / EVAL_MARKER).exists(),
            'lr': self.opt.param_groups[0]['lr'],
            'gpu_mem_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0,
        }
        rec.update({k: float(v) for k, v in kv.items() if k not in ('step', 'samples')})
        rec['loss_by_t'] = _by_t.pop()
        if self.args.use_ema:
            rec['ema_rel_dist'] = self.ema_rel_dist()
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        for k in ['steps_per_sec', 'step_ms', 'data_ms', 'gpu_mem_gb', 'ema_rel_dist']:
            if k in rec:
                self.train_platform.report_scalar(name=k, value=rec[k], iteration=step, group_name='Run')
        for q in range(4):
            if f'loss_q{q}' in kv:
                self.train_platform.report_scalar(name=f'loss_q{q}', value=float(kv[f'loss_q{q}']), iteration=step,
                                                  group_name='LossByQuartile')
        self.acc = {'step_sec': 0.0, 'data_sec': 0.0, 'n': 0}
        self.t_interval = now


if __name__ == '__main__':
    stock_loop.log_loss_dict = log_loss_dict
    stock.TrainLoop = InstrumentedTrainLoop
    try:
        stock.main()
    except StopTraining:
        print('Reached num_steps, stopping.')
