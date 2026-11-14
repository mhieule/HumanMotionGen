"""Learning curves and proxy-length analysis of a training run. Re-run any time during the run.

    cd baselines/mdm && .venv/bin/python ../../experiments/baseline/analyze.py --run save/E003_mdm50_baseline

Reads <run>/train_log.jsonl and <run>/eval/{val,test,val_loss}/*.json (written by
train_instrumented.py and watch.py); writes RESULTS.md and plots/*.png next to this file.
"Final" means the mean of the last FINAL_N evaluated checkpoints; while the run is still going
this is provisional and every number that depends on it is marked so.

Proxy-length criterion (decided by Hieu, 2026-11-14): a training length is good enough at tolerance
X when FID, R-precision top-3 and MM-Dist are each within X of their final values (relative
difference), reported at two levels, X = 5% and 10%. "Reached" = the earliest checkpoint from which
every later evaluated checkpoint stays within X on all three metrics (so one lucky dip of a noisy
FID does not count).
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FINAL_N = 3
CANDIDATES = [25_000, 50_000, 100_000, 150_000, 200_000, 300_000, 400_000]
TOLERANCES = [0.05, 0.10]                 # proxy-length levels: relative distance to the final value
PROXY_METRICS = [('fid', 'FID'), ('r3', 'R-top3'), ('matching', 'MM-Dist')]
COLOR = {'raw': '#2a78d6', 'ema': '#eb6834'}   # categorical slots 1-2 of the dataviz reference palette
LABEL = {'raw': 'raw weights', 'ema': 'EMA weights'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
# key in the eval JSON, column title, better direction, digits
METRICS = [('fid', 'FID', 'lower', 3), ('r3', 'R-precision top-3', 'higher', 3), ('matching', 'MM-Dist', 'lower', 3),
           ('diversity', 'Diversity', 'real', 2), ('tmr_sim', 'TMR similarity', 'higher', 3),
           ('slide_cms', 'Foot sliding (cm/s)', 'lower', 1)]


def load_eval(path):
    r = json.loads(path.read_text())
    if not r.get('complete'):
        return None
    out = {}
    for k, v in r['metrics'].items():
        if k == 'r_precision':
            for i in range(3):
                out[f'r{i + 1}'] = (v['mean'][i], v['ci95'][i])
        elif k == 'gt_r_precision':
            out['gt_r3'] = (v['mean'][2], v['ci95'][2])
        else:
            out[k] = (v['mean'], v['ci95'])
    out['reps'] = r['reps_done']
    return out


def load_run(run):
    ev = {}  # (split, weights, sampler) -> {step: metrics}
    for split in ['val', 'test']:
        for p in sorted((run / 'eval' / split).glob('s*.json')):
            step, w, sampler = p.stem.split('_')
            m = load_eval(p)
            if m:
                ev.setdefault((split, w, sampler), {})[int(step[1:])] = m
    loss = {int(p.stem[1:]): json.loads(p.read_text()) for p in sorted((run / 'eval' / 'val_loss').glob('s*.json'))}
    log = [json.loads(l) for l in (run / 'train_log.jsonl').read_text().splitlines() if l]
    return ev, loss, log


def reference(run):
    """Released MDM-50 checkpoint and real motions on val, re-evaluated with the attention-kernel fix (same evaluator, sampler and seeds as E002)."""
    p = run.parent / 'tier0_fixed' / 'val' / 'cfg2.5.json'
    if not p.exists():
        return {}, {}
    m = load_eval(p)
    real = {'r3': m['gt_r3'][0], 'matching': m['gt_matching'][0], 'diversity': m['gt_diversity'][0], 'fid': None}
    return {k: m[k][0] for k, *_ in METRICS if k in m}, real


def style(ax, title):
    ax.set_title(title, loc='left', fontsize=10, color=INK, pad=6)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def step_axis(ax, steps):
    ax.set_xscale('log')
    ticks = [t for t in [5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000, 750_000]
             if min(steps) <= t <= max(steps)] or [min(steps)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t // 1000}k' for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel('training steps (log scale)', fontsize=8, color=MUTED)


def plot_metrics(ev, ref, real, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (key, title, direction, _) in zip(axes.flat, METRICS):
        style(ax, f'{title}  ({ {"lower": "lower is better", "higher": "higher is better", "real": "closer to real is better"}[direction]})')
        steps_all = []
        for w in ['raw', 'ema']:
            d = ev.get(('val', w, 'cfg2.5'), {})
            steps = sorted(s for s in d if key in d[s])
            if not steps:
                continue
            steps_all += steps
            y = np.array([d[s][key][0] for s in steps])
            ci = np.array([d[s][key][1] for s in steps])
            ax.fill_between(steps, y - ci, y + ci, color=COLOR[w], alpha=0.15, linewidth=0)
            ax.plot(steps, y, color=COLOR[w], linewidth=2, marker='o', markersize=3.5, label=LABEL[w])
        if not steps_all:
            continue
        step_axis(ax, steps_all)
        if key == 'fid':
            ax.set_yscale('log')
        else:
            ax.ticklabel_format(axis='y', useOffset=False)
        if ref.get(key) is not None:
            ax.axhline(ref[key], color=MUTED, linewidth=1.2, linestyle='--', label='released checkpoint (750k)')
        if real.get(key) is not None:
            ax.axhline(real[key], color=MUTED, linewidth=1.2, linestyle=':', label='real motions')
        if key != 'fid':  # keep the plateau readable: clip the first, far-off checkpoints
            vals = [d[s][key][0] for (sp, _, sm), d in ev.items() if sp == 'val' and sm == 'cfg2.5' for s in d if s >= 25_000]
            vals += [v for v in (ref.get(key), real.get(key)) if v is not None]
            if len(vals) > 2:
                lo, hi = min(vals), max(vals)
                ax.set_ylim(lo - 0.15 * (hi - lo + 1e-9), hi + 0.15 * (hi - lo + 1e-9))
    handles = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc='upper right', ncol=len(handles), fontsize=9, frameon=False, labelcolor=MUTED)
    fig.suptitle('Benchmark metrics on val along training (CFG 2.5; band = 95% CI)',
                 x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_loss(loss, log, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    style(ax, 'Training loss (2k-step average)')
    steps = np.array([r['step'] for r in log]); y = np.array([r['loss'] for r in log])
    k = 20
    if len(y) > 2 * k:
        ys = np.convolve(y, np.ones(k) / k, mode='valid'); xs = steps[k - 1:]
        ax.plot(xs[xs >= 2000], ys[xs >= 2000], color=COLOR['raw'], linewidth=2)
        step_axis(ax, [max(2000, int(xs[0])), int(xs[-1])])
    ax = axes[1]
    style(ax, 'Held-out loss, mean over timesteps')
    ls = sorted(loss)
    if ls:
        for w in ['raw', 'ema']:
            if w in loss[ls[0]]:
                ax.plot(ls, [np.mean(loss[s][w]['val_cond']) for s in ls], color=COLOR[w], linewidth=2, marker='o',
                        markersize=3.5, label=f'val, {LABEL[w]}')
        ax.plot(ls, [np.mean(loss[s]['raw']['train_cond']) for s in ls], color=COLOR['raw'], linewidth=1.5,
                linestyle='--', label='train subset, raw weights')
        step_axis(ax, ls)
        ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)
        ax = axes[2]
        style(ax, 'Held-out val loss by timestep (raw weights)')
        ts = loss[ls[0]]['timesteps']
        picks = sorted({ls[0], ls[len(ls) // 4], ls[len(ls) // 2], ls[-1]})
        greys = np.linspace(0.75, 0.0, len(picks))
        for s, g in zip(picks, greys):
            ax.plot(ts, loss[s]['raw']['val_cond'], color=str(g), linewidth=2, marker='o', markersize=3.5,
                    label=f'{s // 1000}k steps')
        ax.set_xlabel('diffusion timestep (0 = almost clean, 49 = pure noise)', fontsize=8, color=MUTED)
        ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fmt(v, digits, ci=None):
    return '' if v is None else f'{v:.{digits}f}' + (f' ± {ci:.{digits}f}' if ci is not None else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    a = ap.parse_args()
    ev, loss, log = load_run(a.run)
    ref, real = reference(a.run)
    args = json.loads((a.run / 'args.json').read_text())
    (HERE / 'plots').mkdir(exist_ok=True)
    plot_metrics(ev, ref, real, HERE / 'plots' / 'metrics_vs_steps.png')
    plot_loss(loss, log, HERE / 'plots' / 'loss.png')

    free = [r['steps_per_sec'] for r in log if not r['eval_running'] and r['step'] > 0]
    speed = float(np.median(free)) if free else float('nan')   # steps/s with the GPU to itself
    hours = lambda s: s / speed / 3600
    last = log[-1]['step']
    finished = last >= args['num_steps']
    prov = '' if finished else ' (provisional)'
    status = 'complete' if finished else f'**running**, step {last:,} of {args["num_steps"]:,}'
    L = [f'# E003 results: MDM-50 baseline, learning curves and proxy length',
         '',
         f'_Generated by `experiments/baseline/analyze.py`, {time.strftime("%Y-%m-%d %H:%M")}. '
         f'Status: {status}._',
         '',
         f'Run: `{a.run}`. Recipe: released MDM-50 config, seed {args["seed"]}; raw weights are the reproduction, '
         'EMA (0.9999) is tracked alongside. Benchmark: Guo et al. evaluator on **val**, 1000 motions × 3 replications, '
         'MDM sampler (CFG 2.5), mean ± 95% CI. Design: `docs/superpowers/specs/2026-11-14-mdm-baseline-run-design.md`.',
         '',
         f'Speed with the GPU to itself: **{speed:.2f} steps/s** (median of log intervals with no evaluation running); '
         f'the hours below are steps ÷ this speed, i.e. pure training time. '
         f'Wall-clock so far: {log[-1]["train_sec"] / 3600:.1f} h including shared-GPU evaluation.',
         '',
         '![metrics](plots/metrics_vs_steps.png)',
         '',
         '![loss](plots/loss.png)',
         '']

    main_ev = {w: ev.get(('val', w, 'cfg2.5'), {}) for w in ['raw', 'ema']}
    for w in ['raw', 'ema']:
        d = main_ev[w]
        if len(d) < FINAL_N + 1:
            continue
        steps = sorted(d)
        fin = {k: float(np.mean([d[s][k][0] for s in steps[-FINAL_N:]])) for k, *_ in METRICS}
        fin_ci = {k: float(np.mean([d[s][k][1] for s in steps[-FINAL_N:]])) for k, *_ in METRICS}
        gap = lambda s, k: abs(d[s][k][0] - fin[k]) / abs(fin[k])          # relative distance to final
        worst = lambda s: max(gap(s, k) for k, _ in PROXY_METRICS)
        reached = {}
        for tol in TOLERANCES:   # earliest checkpoint from which all later ones stay within tol
            ok_from = None
            for st in steps:
                if worst(st) <= tol:
                    ok_from = st if ok_from is None else ok_from
                else:
                    ok_from = None
            reached[tol] = ok_from
        lvl = ' / '.join(f'{t:.0%}' for t in TOLERANCES)
        L += [f'## Proxy length, {LABEL[w]}{prov}', '',
              f'Final = mean of the last {FINAL_N} evaluated checkpoints ({", ".join(f"{s // 1000}k" for s in steps[-FINAL_N:])}): '
              f'FID {fin["fid"]:.3f}, R-top3 {fin["r3"]:.3f}, MM-Dist {fin["matching"]:.3f}. '
              f'Gaps are relative to final; a length passes a level when all three gaps are within it.', '']
        for tol in TOLERANCES:
            r = reached[tol]
            L.append(f'- **Within {tol:.0%} of final on all three, and staying there: '
                     f'{f"from {r // 1000}k steps ({hours(r):.1f} h of training)" if r else "not reached yet"}**')
        noise = {k: np.mean([d[s][k][1] for s in steps[len(steps) // 2:]]) / abs(fin[k]) for k, _ in PROXY_METRICS}
        L += [f'- Measurement noise: one evaluation\'s 95% CI is ±{noise["fid"]:.0%} of final for FID, '
              f'±{noise["r3"]:.1%} for R-top3, ±{noise["matching"]:.1%} for MM-Dist (mean over the second half of the run). '
              'Where this is close to a tolerance, single checkpoints can pass or fail by chance; '
              'the candidates around the chosen length get 10 replications after the run.', '',
              '| Proxy length | Training time | FID | gap | R-top3 | gap | MM-Dist | gap | Passes ' + lvl + ' |',
              '|---|---|---|---|---|---|---|---|---|']
        for c in CANDIDATES:
            if c not in d:
                continue
            m = d[c]
            passes = ' / '.join('yes' if worst(c) <= t else 'no' for t in TOLERANCES)
            L.append(f'| {c // 1000}k | {hours(c):.1f} h | {fmt(m["fid"][0], 3, m["fid"][1])} | {gap(c, "fid"):.0%} | '
                     f'{m["r3"][0]:.3f} | {gap(c, "r3"):.1%} | {m["matching"][0]:.3f} | {gap(c, "matching"):.1%} | {passes} |')
        if len(steps) > 6:
            tail = [d[s]['fid'][0] for s in steps[len(steps) // 2:]]
            L += ['', f'Checkpoint-to-checkpoint spread of FID over the second half of the evaluated checkpoints: '
                      f'std {np.std(tail):.3f} ({np.std(tail) / fin["fid"]:.0%} of final).']
        L.append('')

    L += ['## All evaluated checkpoints (val, CFG 2.5)', '',
          '| Step | Train time | Weights | FID ↓ | R@1 ↑ | R@3 ↑ | MM-Dist ↓ | Diversity → | TMR sim ↑ | Foot slide cm/s ↓ | Floating % ↓ | Held-out loss val / train |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for s in sorted(set(main_ev['raw']) | set(main_ev['ema'])):
        for w in ['raw', 'ema']:
            m = main_ev[w].get(s)
            if not m:
                continue
            hl = loss.get(s, {}).get(w)
            hl = f'{np.mean(hl["val_cond"]):.4f} / {np.mean(hl["train_cond"]):.4f}' if hl else ''
            L.append(f'| {s // 1000}k | {hours(s):.1f} h | {w} | {fmt(m["fid"][0], 3, m["fid"][1])} | {m["r1"][0]:.3f} | '
                     f'{fmt(m["r3"][0], 3, m["r3"][1])} | {m["matching"][0]:.3f} | {m["diversity"][0]:.2f} | '
                     f'{m["tmr_sim"][0]:.3f} | {m["slide_cms"][0]:.1f} | {m["float_pct"][0]:.1f} | {hl} |')
    if ref:
        L.append(f'| *released, 750k* | | raw | {ref["fid"]:.3f} | | {ref["r3"]:.3f} | {ref["matching"]:.3f} | '
                 f'{ref["diversity"]:.2f} | {ref["tmr_sim"]:.3f} | {ref["slide_cms"]:.1f} | | |')
    if real:
        L.append(f'| *real motions* | | | ~0 | | {real["r3"]:.3f} | {real["matching"]:.3f} | {real["diversity"]:.2f} | | | | |')

    extra = [(k, d) for k, d in sorted(ev.items()) if k[2] != 'cfg2.5' or k[0] == 'test']
    if extra:
        L += ['', '## Other evaluations', '',
              '| Split | Sampler | Weights | Step | Reps | FID ↓ | R@3 ↑ | MM-Dist ↓ | Diversity → |', '|---|---|---|---|---|---|---|---|---|']
        for (split, w, sampler), d in extra:
            for s in sorted(d):
                m = d[s]
                L.append(f'| {split} | {sampler} | {w} | {s // 1000}k | {m["reps"]} | {fmt(m["fid"][0], 3, m["fid"][1])} | '
                         f'{fmt(m["r3"][0], 3, m["r3"][1])} | {m["matching"][0]:.3f} | {m["diversity"][0]:.2f} |')
    (HERE / 'RESULTS.md').write_text('\n'.join(L) + '\n')
    n = sum(len(d) for d in ev.values())
    print(f'step {last:,}, {speed:.2f} steps/s uncontended, {n} benchmark evaluations, {len(loss)} loss evaluations '
          f'-> {HERE / "RESULTS.md"}')


if __name__ == '__main__':
    main()
