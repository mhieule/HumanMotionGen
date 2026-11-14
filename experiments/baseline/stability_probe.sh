#!/usr/bin/env bash
# Does the first part of MDM training diverge? Short runs of the released recipe for several
# seeds, with and without extra flags, logged every 10 steps. Run from baselines/mdm:
#     bash ../../experiments/baseline/stability_probe.sh <name> "<extra train flags>" [steps] [seeds]
#     e.g.  ... nomask ""      and      ... mask "--mask_frames"
# LAUNCHER=diag_spike.py capture reproduces the 2026-11-14 divergence (that script leaves the faulty kernel on).
# Then: .venv/bin/python ../../experiments/baseline/stability_report.py save/_diag
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
name=$1; extra=${2:-}; steps=${3:-2500}; seeds=${4:-"10 11 12"}
export TQDM_MININTERVAL=30
for s in $seeds; do
    d=save/_diag/${name}_s$s
    [ -f $d/model$(printf %09d $steps).pt ] && continue
    .venv/bin/python "$here/train_instrumented.py" --save_dir $d --overwrite --dataset humanml --diffusion_steps 50 \
        --seed $s --num_steps $steps --save_interval $steps --log_interval 10 $extra > $d.log 2>&1
done
