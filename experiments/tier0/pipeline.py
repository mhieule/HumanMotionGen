"""Tier 0 pipeline: sweep sampling configs on val, pick winners, compare to MDM on test.

Run from baselines/mdm with its venv (or via run_tier0.sh):
    python ../../experiments/tier0/pipeline.py [--stage1_reps 5] [--stage2_reps 20]

Stage 1 (val, selection): CFG scale sweep, interval CFG, APG, best-of-N.
Selection: per family, lowest val FID among configs whose R-precision top-3 is not worse than
the baseline's (mean >= baseline mean - baseline 95% CI). Then the best sampler overall is
combined with the best best-of-N verifier.
Stage 2 (test, standard 20 reps): baseline (MDM default, CFG 2.5) + the winners.
Resumable: a config whose JSON says complete is skipped. Writes RESULTS.md next to this file.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = 'save/humanml_enc_512_50steps/model000750000.pt'
OUT = Path('save/tier0')
EXTRA = ''

# name -> eval_tier0.py flags. t indices: 0 = clean ... 49 = pure noise. Interval CFG paper's
# sigma range (0.28, 5.42] is t 8..43 for MDM's 50-step cosine schedule.
BASE = ('cfg2.5', '--scale 2.5')
STAGE1 = dict([BASE] + [
    (f'cfg{s}', f'--scale {s}') for s in [1.5, 2.0, 3.0, 3.5, 4.0, 5.0]] + [
    ('int8-43_s2.5', '--scale 2.5 --interval 8 43'),
    ('int8-43_s3.5', '--scale 3.5 --interval 8 43'),
    ('int8-43_s5', '--scale 5 --interval 8 43'),
    ('int14-40_s3.5', '--scale 3.5 --interval 14 40'),
    ('int14-40_s5', '--scale 5 --interval 14 40'),
    ('int8-49_s3.5', '--scale 3.5 --interval 8 49'),    # skip only the lowest-noise steps
    ('int0-43_s3.5', '--scale 3.5 --interval 0 43'),    # skip only the highest-noise steps
    ('apg_s2.5', '--scale 2.5 --apg_eta 0'),
    ('apg_s4', '--scale 4 --apg_eta 0'),
    ('apg_s6', '--scale 6 --apg_eta 0'),
    ('apg_s4_m-.5', '--scale 4 --apg_eta 0 --apg_momentum -0.5'),
    ('apg_s6_m-.5', '--scale 6 --apg_eta 0 --apg_momentum -0.5'),
    ('apg_s8_m-.5', '--scale 8 --apg_eta 0 --apg_momentum -0.5'),
    ('apg_s4_eta.5', '--scale 4 --apg_eta 0.5'),
    ('bon4_tmr', '--scale 2.5 --best_of 4 --verifier tmr'),
    ('bon4_tmrphys', '--scale 2.5 --best_of 4 --verifier tmr_phys'),
    ('bon8_tmrphys', '--scale 2.5 --best_of 8 --verifier tmr_phys'),
])
FAMILY = lambda n: 'combined' if '+bon' in n else 'best-of-N' if n.startswith('bon') else 'interval' if n.startswith('int') else \
    'APG' if n.startswith('apg') else 'CFG scale'


def run(name, flags, split, reps, extra=''):
    out = OUT / split
    js = out / f'{name}.json'
    if js.exists() and json.loads(js.read_text()).get('complete') and json.loads(js.read_text())['reps_done'] >= reps:
        print(f'[skip] {split}/{name}', flush=True)
        return
    cmd = (f'{sys.executable} {HERE / "eval_tier0.py"} --model_path {MODEL} --out_dir {out} --split {split} '
           f'--reps {reps} --name {name} {flags} {extra} {EXTRA}')
    print(f'[run]  {split}/{name}: {flags}  ({time.strftime("%H:%M")})', flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    (out / f'{name}.stderr').write_text(p.stderr[-20000:])
    last = [l for l in p.stdout.splitlines() if l.startswith(name + ':')]
    print(f'       {last[-1] if last else "FAILED (see " + str(out / (name + ".stderr")) + ")"}'
          f'  [{(time.time() - t0) / 60:.0f} min]', flush=True)


def load(split):
    res = {}
    for js in sorted((OUT / split).glob('*.json')):
        r = json.loads(js.read_text())
        if r.get('metrics'):
            res[r['name']] = r
    return res


def m(r, k, i=None):
    v = r['metrics'][k]
    return (v['mean'][i], v['ci95'][i]) if i is not None else (v['mean'], v['ci95'])


def select(res):
    base = res[BASE[0]]
    floor = m(base, 'r_precision', 2)[0] - m(base, 'r_precision', 2)[1]
    ok = {n: r for n, r in res.items() if r.get('complete') and m(r, 'r_precision', 2)[0] >= floor}
    best = {}
    for n, r in ok.items():
        f = FAMILY(n)
        if f not in best or m(r, 'fid')[0] < m(ok[best[f]], 'fid')[0]:
            best[f] = n
    return best, floor


def table(res, base_name, title, note=''):
    base = res.get(base_name)
    hdr = ('| Config | Family | Reps | FID ↓ | R@1 ↑ | R@3 ↑ | MM-Dist ↓ | Diversity → | TMR sim ↑ | '
           'Foot slide cm/s ↓ | Floating % ↓ | Jerk ↓ | NFE / motion | Sampling time |')
    rows = [f'### {title}', note, '', hdr, '|' + '---|' * 14]
    for n, r in sorted(res.items(), key=lambda kv: m(kv[1], 'fid')[0]):
        sec = m(r, 'sample_sec')[0]
        rel = f'{sec:.0f} s/rep' + (f' ({sec / m(base, "sample_sec")[0]:.1f}×)' if base else '')
        star = ' **(MDM default)**' if n == base_name else ''
        rows.append(
            f'| {n}{star} | {FAMILY(n)} | {r["reps_done"]} | {m(r, "fid")[0]:.3f} ± {m(r, "fid")[1]:.3f} | '
            f'{m(r, "r_precision", 0)[0]:.3f} | {m(r, "r_precision", 2)[0]:.3f} ± {m(r, "r_precision", 2)[1]:.3f} | '
            f'{m(r, "matching")[0]:.3f} | {m(r, "diversity")[0]:.3f} | {m(r, "tmr_sim")[0]:.3f} | '
            f'{m(r, "slide_cms")[0]:.1f} | {m(r, "float_pct")[0]:.1f} | {m(r, "jerk")[0]:.0f} | '
            f'{m(r, "nfe_per_motion")[0]:.0f} | {rel} |')
    gt = base.get('gt_extras') if base else None
    if base:
        gr = (base['metrics']['gt_r_precision']['mean'], base['metrics']['gt_matching']['mean'],
              base['metrics']['gt_diversity']['mean'])
        rows.append(f'| *Real motions* | | | ~0 | {gr[0][0]:.3f} | {gr[0][2]:.3f} | {gr[1]:.3f} | '
                    f'{gr[2]:.3f} | {gt["tmr_sim"]:.3f} | {gt["slide_cms"]:.1f} | {gt["float_pct"]:.1f} | '
                    f'{gt["jerk"]:.0f} | | |' if gt else '')
    return '\n'.join(rows) + '\n'


def write_results(stage1, stage2, best, floor, combo, status):
    txt = [
        '# Tier 0 results: inference-time guidance on pretrained MDM-50', '',
        f'_Status: {status}. Generated by `experiments/tier0/pipeline.py`, {time.strftime("%Y-%m-%d %H:%M")}._', '',
        'Model: `save/humanml_enc_512_50steps/model000750000.pt` (released MDM, 50 diffusion steps, no retraining). '
        'Metrics: MDM\'s HumanML3D evaluator (Guo et al., epoch-28 matcher), 1000 generated motions per replication, '
        'mean ± 95% CI over replications. Replication k uses seed 10+k for every config (same prompts per replication). '
        'TMR sim = cosine similarity of TMR text/motion latents (not the benchmark evaluator). Physics: '
        '`scripts/motion_stats.py` on recovered joints. NFE = network passes per final motion.', '',
    ]
    if stage2:
        txt.append(table(stage2, BASE[0], 'Stage 2: final comparison on the test split',
                         'Selected on val (stage 1), evaluated on test. This is the table to report.'))
    if best:
        txt.append(f'**Selected on val** (lowest FID per family with R@3 ≥ {floor:.3f} = baseline − CI): ' +
                   ', '.join(f'{f}: `{n}`' for f, n in best.items()) + (f'; combined: `{combo[0]}` ({combo[1]})' if combo else '') + '\n')
    if stage1:
        txt.append(table(stage1, BASE[0], 'Stage 1: sweep on the val split (selection only)',
                         'Fewer replications; CIs are wider. Do not report these as final numbers.'))
    (HERE / 'RESULTS.md').write_text('\n'.join(txt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1_reps', type=int, default=5)
    ap.add_argument('--stage2_reps', type=int, default=20)
    ap.add_argument('--quick', action='store_true', help='pipeline test: 5 configs, 96 motions, 1 rep, save/tier0_quick')
    a = ap.parse_args()
    global OUT, EXTRA
    if a.quick:
        OUT, EXTRA = Path('save/tier0_quick'), '--num_samples 96'
        a.stage1_reps = a.stage2_reps = 1
        for n in list(STAGE1):
            if n not in [BASE[0], 'cfg3.0', 'int8-43_s3.5', 'apg_s4', 'bon4_tmrphys']:
                del STAGE1[n]

    print('== Stage 1: val sweep', flush=True)
    run(*BASE, 'val', a.stage1_reps, '--gt_reference')
    for n, f in STAGE1.items():
        if n != BASE[0]:
            run(n, f, 'val', a.stage1_reps)
            write_results(load('val'), {}, None, 0, None, 'stage 1 running')

    s1 = load('val')
    best, floor = select(s1)
    samplers = [best[f] for f in ['CFG scale', 'interval', 'APG'] if f in best]
    best_sampler = min(samplers, key=lambda n: m(s1[n], 'fid')[0])
    combo = None
    if 'best-of-N' in best:
        verifier = STAGE1[best['best-of-N']].split('--verifier ')[1].split()[0]
        # stage 2 caps best-of-N at N=4 (N=8 x 20 reps would take ~10 h)
        combo_name = f'{best_sampler}+bon4_{verifier.replace("_", "")}'
        combo = (combo_name, f'{STAGE1[best_sampler]} --best_of 4 --verifier {verifier}')
    print(f'== Selected: {best}; best sampler {best_sampler}; combined {combo}', flush=True)
    (OUT / 'selection.json').write_text(json.dumps({'best': best, 'floor': floor, 'best_sampler': best_sampler,
                                                   'combo': combo}, indent=1))

    print('== Stage 2: test, final comparison', flush=True)
    order = [BASE[0], best_sampler] + [n for n in samplers if n != best_sampler]
    run(*BASE, 'test', a.stage2_reps, '--gt_reference')
    for n in order[1:]:
        if n != BASE[0]:
            run(n, STAGE1[n], 'test', a.stage2_reps)
            write_results(s1, load('test'), best, floor, combo, 'stage 2 running')
    if combo:
        run(*combo, 'test', a.stage2_reps)
    write_results(s1, load('test'), best, floor, combo, 'complete' + (' (QUICK TEST, not real numbers)' if a.quick else ''))
    print('== DONE', flush=True)
    (OUT / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M'))


if __name__ == '__main__':
    main()
