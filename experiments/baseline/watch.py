"""Evaluate the checkpoints of a training run as they appear. Separate process from training.

Run from baselines/mdm with its venv:
    python ../../experiments/baseline/watch.py --run save/E003_mdm50_baseline --num_steps 750000

Jobs, by priority (lower first; within a priority, earliest checkpoint first):
  0  held-out denoising loss (val_loss.py)        every 5k to 100k, then the benchmark schedule
  1  benchmark on val, MDM sampler (CFG 2.5)       every 5k to 50k, 10k to 200k, 25k to the end
  1  final numbers on test, 20 replications        steps 600k and the last step
  2  benchmark on val, APG scale 4 (E002 sampler)  every 50k
Benchmark jobs run for raw and EMA weights separately (experiments/tier0/eval_tier0.py).
Outputs: <run>/eval/{val_loss,val,test}/*.json. Resumable: finished jobs are skipped, a job
that fails twice is listed in <run>/eval/failed.json and skipped. While a job runs,
<run>/eval/RUNNING exists so the training log can mark GPU-shared intervals.
Every finished job is also written to TensorBoard under <run>/eval/tb/<weights>_<sampler>[_test],
with the same tag names for raw and EMA weights, so they overlay in one chart.
Exits when the last checkpoint exists and nothing is pending. `--list` prints the queue.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIER0 = HERE.parent / 'tier0' / 'eval_tier0.py'

SAMPLERS = {'cfg2.5': '--scale 2.5', 'apg4': '--scale 4 --apg_eta 0'}


def schedule(num_steps):
    main = sorted(set(list(range(5_000, 50_001, 5_000)) + list(range(60_000, 200_001, 10_000))
                      + list(range(225_000, num_steps + 1, 25_000)) + [num_steps]))
    loss = sorted(set(list(range(5_000, 100_001, 5_000)) + main))
    apg = list(range(50_000, num_steps + 1, 50_000))
    final = sorted({s for s in (600_000, num_steps) if s <= num_steps})
    return main, loss, apg, final


def all_jobs(run, num_steps, val_reps, test_reps, has_ema):
    main, loss, apg, final = schedule(num_steps)
    py, ev = sys.executable, run / 'eval'
    weights = ['raw', 'ema'] if has_ema else ['raw']
    jobs = []

    def bench(prio, step, split, sampler, reps, w):
        name = f's{step:09d}_{w}_{sampler}'
        jobs.append(dict(prio=prio, step=step, name=f'{split}/{name}', out=ev / split / f'{name}.json',
                         cmd=f'{py} {TIER0} --model_path {run}/model{step:09d}.pt --out_dir {ev / split} --name {name} '
                             f'--split {split} --reps {reps} --weights {w} {SAMPLERS[sampler]}'))
    for s in loss:
        jobs.append(dict(prio=0, step=s, name=f'val_loss/s{s:09d}', out=ev / 'val_loss' / f's{s:09d}.json',
                         cmd=f'{py} {HERE / "val_loss.py"} --model_path {run}/model{s:09d}.pt '
                             f'--out {ev / "val_loss" / f"s{s:09d}.json"}'))
    for s in main:
        for w in weights:
            bench(1, s, 'val', 'cfg2.5', val_reps, w)
    for s in final:
        for w in weights:
            bench(1, s, 'test', 'cfg2.5', test_reps, w)
    for s in apg:
        for w in weights:
            bench(2, s, 'val', 'apg4', val_reps, w)
    return sorted(jobs, key=lambda j: (j['prio'], j['step']))


def to_tensorboard(ev, job):
    """Append one finished job's numbers to TensorBoard (one TB run per weights/sampler/split)."""
    from torch.utils.tensorboard import SummaryWriter
    r, step = json.loads(job['out'].read_text()), job['step']
    if '/val_loss/' in str(job['out']):
        for w in ['raw', 'ema']:
            if w in r:
                with SummaryWriter(str(ev / 'tb' / f'{w}_cfg2.5')) as tb:
                    for k, v in r[w].items():
                        tb.add_scalar(f'HeldOutLoss/{k}', sum(v) / len(v), step)
                    tb.add_scalar('HeldOutLoss/val_minus_train', (sum(r[w]['val_cond']) - sum(r[w]['train_cond']))
                                  / len(r[w]['val_cond']), step)
        return
    split, name = job['name'].split('/')
    _, w, sampler = name.split('_')
    m = {k: v['mean'] for k, v in r['metrics'].items()}
    with SummaryWriter(str(ev / 'tb' / (f'{w}_{sampler}' + ('' if split == 'val' else f'_{split}')))) as tb:
        for i in range(3):
            tb.add_scalar(f'Eval/r_precision_top{i + 1}', m['r_precision'][i], step)
        for tag, k in [('Eval/fid', 'fid'), ('Eval/mm_dist', 'matching'), ('Eval/diversity', 'diversity'),
                       ('Eval/tmr_sim', 'tmr_sim'), ('Physics/foot_slide_cm_s', 'slide_cms'),
                       ('Physics/floating_pct', 'float_pct'), ('Physics/jerk', 'jerk'),
                       ('Eval/fid_ci95', None)]:
            tb.add_scalar(tag, r['metrics']['fid']['ci95'] if k is None else m[k], step)


def done(job):
    if not job['out'].exists():
        return False
    if '/val_loss/' in str(job['out']):
        return True
    return bool(json.loads(job['out'].read_text()).get('complete'))


def ready(run, step):
    """A checkpoint is complete once the optimizer file (written after it) exists and has settled."""
    m, o = run / f'model{step:09d}.pt', run / f'opt{step:09d}.pt'
    return m.exists() and o.exists() and time.time() - o.stat().st_mtime > 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--num_steps', type=int, required=True)
    ap.add_argument('--val_reps', type=int, default=3)
    ap.add_argument('--test_reps', type=int, default=20)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--poll', type=int, default=60)
    a = ap.parse_args()
    ev = a.run / 'eval'
    (ev / 'logs').mkdir(parents=True, exist_ok=True)
    has_ema = json.loads((a.run / 'args.json').read_text())['use_ema']
    jobs = all_jobs(a.run, a.num_steps, a.val_reps, a.test_reps, has_ema)
    failed_path, marker = ev / 'failed.json', ev / 'RUNNING'
    failed = json.loads(failed_path.read_text()) if failed_path.exists() else {}
    marker.unlink(missing_ok=True)

    if a.list:
        for j in jobs:
            state = 'done' if done(j) else 'FAILED' if failed.get(j['name'], 0) >= 2 else \
                'pending' if ready(a.run, j['step']) else 'waiting for checkpoint'
            print(f"p{j['prio']} {j['name']:40s} {state}")
        return

    print(f'{len(jobs)} jobs for {a.run} ({time.strftime("%Y-%m-%d %H:%M")})', flush=True)
    while True:
        todo = [j for j in jobs if not done(j) and failed.get(j['name'], 0) < 2]
        if not todo:
            print('all jobs done', flush=True)
            return
        runnable = [j for j in todo if ready(a.run, j['step'])]
        if not runnable:
            time.sleep(a.poll)
            continue
        j = runnable[0]
        log = ev / 'logs' / (j['name'].replace('/', '_') + '.log')
        marker.write_text(j['name'])
        t0 = time.time()
        with open(log, 'a') as f:
            code = subprocess.run(j['cmd'], shell=True, stdout=f, stderr=subprocess.STDOUT).returncode
        marker.unlink(missing_ok=True)
        ok = code == 0 and done(j)
        if ok:
            try:
                to_tensorboard(ev, j)
            except Exception as e:  # tracking must never stop the queue
                print(f'TensorBoard write failed for {j["name"]}: {e!r}', flush=True)
        if not ok:
            failed[j['name']] = failed.get(j['name'], 0) + 1
            failed_path.write_text(json.dumps(failed, indent=1))
        print(f"{time.strftime('%m-%d %H:%M')} {'ok    ' if ok else 'FAILED'} {j['name']:36s} "
              f"{(time.time() - t0) / 60:5.1f} min  ({len(todo) - ok} left)", flush=True)


if __name__ == '__main__':
    main()
