"""Physical-plausibility stats for MDM samples, with real HumanML3D clips as reference.

Usage (inside the container, from /mdm):
    python scripts/motion_stats.py <samples_dir> [--gt-words walk circle] [--gt-max 30]

Reads <samples_dir>/results.npy (joints, y-up, meters, 20 fps), prints one row per
sample plus the mean over matching test-set clips, and writes <samples_dir>/motion_stats.png
with the top-down root paths.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FPS = 20
ROOT, FEET = 0, [10, 11]          # pelvis; left/right foot (toe-base) joints
CONTACT_H = 0.05                  # foot counts as planted below 5 cm ...
CONTACT_V = 0.5                   # ... unless it moves faster than 0.5 m/s (swing phase)


def stats(j):
    """j: (frames, 22, 3) joint positions. Returns a dict of scalar metrics."""
    root = j[:, ROOT][:, [0, 2]]                       # horizontal pelvis path
    step = np.diff(root, axis=0)
    speed = np.linalg.norm(step, axis=1) * FPS
    path_len = np.linalg.norm(step, axis=1).sum()

    # Accumulated turning of the walking direction, only while actually moving.
    moving = speed[:-1] > 0.2
    heading = np.unwrap(np.arctan2(step[:, 1], step[:, 0]))
    turn = np.degrees(np.diff(heading)[moving].sum())

    # Least-squares circle through the path (Kasa fit): radius and fit error.
    x, z = root[:, 0], root[:, 1]
    A = np.c_[2 * x, 2 * z, np.ones_like(x)]
    (cx, cz, c), *_ = np.linalg.lstsq(A, x ** 2 + z ** 2, rcond=None)
    radius = np.sqrt(c + cx ** 2 + cz ** 2)
    fit_err = np.abs(np.hypot(x - cx, z - cz) - radius).mean()

    # Ground: y = 0 is the floor in HumanML3D. Lowest foot point per frame.
    low = j[:, [7, 8, 10, 11], 1].min(axis=1)
    float_pct = (low > CONTACT_H).mean() * 100
    sink_cm = np.clip(-low, 0, None).mean() * 100

    # Foot sliding: horizontal foot speed while the foot is planted.
    slide, n_contact = [], 0
    for f in FEET:
        h = j[1:, f, 1]
        v = np.linalg.norm(np.diff(j[:, f][:, [0, 2]], axis=0), axis=1) * FPS
        planted = (h < CONTACT_H) & (v < CONTACT_V)
        slide.append(v[planted]); n_contact += planted.sum()
    slide = np.concatenate(slide)
    slide_cms = slide.mean() * 100 if slide.size else np.nan
    slide_pct = (slide > 0.1).mean() * 100 if slide.size else np.nan   # > 10 cm/s

    # Jitter: mean jerk magnitude over all joints.
    jerk = np.linalg.norm(np.diff(j, n=3, axis=0), axis=2).mean() * FPS ** 3

    return dict(path_m=path_len, turn_deg=turn, closure_m=np.linalg.norm(root[-1] - root[0]),
                radius_m=radius, fit_err_m=fit_err, float_pct=float_pct, sink_cm=sink_cm,
                slide_cms=slide_cms, slide_pct=slide_pct, jerk=jerk), root


COLS = [('path_m', 'path m', '{:6.2f}'), ('turn_deg', 'turn deg', '{:8.0f}'),
        ('closure_m', 'end-start m', '{:11.2f}'), ('radius_m', 'radius m', '{:8.2f}'),
        ('fit_err_m', 'circle err m', '{:12.2f}'), ('float_pct', 'float %', '{:7.0f}'),
        ('sink_cm', 'sink cm', '{:7.1f}'), ('slide_cms', 'slide cm/s', '{:10.1f}'),
        ('slide_pct', 'slide>10 %', '{:10.0f}'), ('jerk', 'jerk m/s3', '{:9.0f}')]


def row(name, s):
    return f'{name:<10}' + ' '.join(fmt.format(s[k]) for k, _, fmt in COLS)


def load_gt(words, limit, root='dataset/HumanML3D'):
    clips = []
    for mid in open(f'{root}/test.txt').read().split():
        if mid.startswith('M'):
            continue
        caps = [l.split('#')[0] for l in open(f'{root}/texts/{mid}.txt')]
        if any(all(w in c.lower() for w in words) for c in caps):
            clips.append((mid, np.load(f'{root}/new_joints/{mid}.npy')))
        if len(clips) == limit:
            break
    return clips


def main():
    p = argparse.ArgumentParser()
    p.add_argument('samples_dir')
    p.add_argument('--gt-words', nargs='+', default=['walk', 'circle'])
    p.add_argument('--gt-max', type=int, default=30)
    a = p.parse_args()

    r = np.load(os.path.join(a.samples_dir, 'results.npy'), allow_pickle=True).item()
    n_rep = r['num_repetitions']
    gen = []
    for i, (m, L) in enumerate(zip(r['motion'], r['lengths'])):
        j = m.transpose(2, 0, 1)[:L]                   # (22, 3, T) -> (T, 22, 3)
        gen.append((f'rep{i // r["num_samples"]}_s{i % r["num_samples"]}', *stats(j)))
    gt = [(mid, *stats(j)) for mid, j in load_gt(a.gt_words, a.gt_max)]

    print(f'Prompt: {r["text"][0]!r}, {len(gen)} samples, {r["lengths"][0] / FPS:.1f} s each')
    print(f'{"":<10}' + ' '.join(h for _, h, _ in COLS))
    for name, s, _ in gen:
        print(row(name, s))
    # Turn is signed (+ = counter-clockwise from above); average its size, and count directions.
    mean = lambda rows: {k: np.nanmean([abs(s[k]) if k == 'turn_deg' else s[k] for _, s, _ in rows])
                         for k, _, _ in COLS}
    ccw = lambda rows: sum(s['turn_deg'] > 0 for _, s, _ in rows)
    print('-' * 110)
    print(row('gen mean', mean(gen)) + f'   ({ccw(gen)}/{len(gen)} turn counter-clockwise)')
    if gt:
        print(row(f'real ({len(gt)})', mean(gt)) + f'   ({ccw(gt)}/{len(gt)} turn counter-clockwise)')
        print(f'real = test-set clips whose caption has all of {a.gt_words}; means use |turn|')

    # Top-down root paths: generated (left) vs real (right), each starting at the origin.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, rows, title in [(axes[0], gen, f'MDM ({len(gen)})'), (axes[1], gt, f'real ({len(gt)})')]:
        for _, _, path in rows:
            ax.plot(*(path - path[0]).T, lw=1)
            ax.plot(*(path[-1] - path[0]), 'k.', ms=4)
        ax.plot(0, 0, 'ko', ms=6)
        ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(title)
        ax.set_xlabel('x (m)'); ax.set_ylabel('z (m)')
    fig.suptitle('Root path from above (big dot = start, small dots = ends)')
    out = os.path.join(a.samples_dir, 'motion_stats.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('plot:', out)


if __name__ == '__main__':
    main()
