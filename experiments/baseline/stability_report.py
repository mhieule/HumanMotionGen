"""Summarise stability_probe.sh runs: where (if at all) each run's loss blows up."""
import json
import sys
from pathlib import Path

import numpy as np

for log in sorted(Path(sys.argv[1]).glob('*/train_log.jsonl')):
    rows = [json.loads(l) for l in log.read_text().splitlines() if l]
    step = np.array([r['step'] for r in rows]); loss = np.array([r['loss'] for r in rows])
    grad = np.array([r['grad_norm'] for r in rows])
    best = np.minimum.accumulate(loss)
    bad = np.where((step > 200) & (loss > 1.5 * best))[0]          # interval mean 50% above the best so far
    spikes = step[grad > 3]                                         # normal gradient norm is ~0.4
    seg = lambda a, b: loss[(step >= a) & (step < b)].mean() if ((step >= a) & (step < b)).any() else float('nan')
    print(f'{log.parent.name:16s} steps {step[-1]:5d} | loss 400-500 {seg(400, 500):.3f}, last 200 {seg(step[-1] - 200, step[-1] + 1):.3f} | '
          f'grad-norm max {grad.max():8.1f}, first interval > 3: {spikes[0] if len(spikes) else "never":>5} ({len(spikes)} intervals) | '
          f'loss 50% above best: {("first at step " + str(step[bad[0]]) + f", {len(bad)} intervals") if len(bad) else "never"}')
