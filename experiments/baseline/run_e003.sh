#!/usr/bin/env bash
# E003: MDM-50 baseline, released recipe, 750k steps. Starts (or resumes) training and the
# checkpoint evaluator in two tmux sessions. Safe to re-run: a session that is alive is left
# alone, training resumes from the newest checkpoint, finished evaluations are skipped.
#     experiments/baseline/run_e003.sh            # start / resume all three sessions
#     SESSIONS="train tb" experiments/baseline/run_e003.sh   # only some of them
#     tmux attach -t e003-train | e003-eval | e003-tb         # watch (detach: Ctrl-b d)
# TensorBoard: http://localhost:6006 on this machine; from elsewhere first
#     ssh -L 6006:localhost:6006 <this machine>
# Design and decisions: docs/superpowers/specs/2026-11-14-mdm-baseline-run-design.md
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mdm=$(cd "$here/../../baselines/mdm" && pwd)
run=save/E003_mdm50_baseline
steps=750000

# Training arguments = the released checkpoint's args.json (all MDM defaults at commit ef8edce
# plus --diffusion_steps 50). --use_ema only keeps an averaged copy next to the raw weights.
# Not in the trained recipe, monitoring only: --save_interval, --log_interval, platform.
train_args="--save_dir $run --overwrite --dataset humanml --diffusion_steps 50 --seed 10 \
--num_steps $steps --save_interval 5000 --log_interval 100 --use_ema \
--train_platform_type TensorboardPlatform"

sessions=${SESSIONS:-train eval tb}
want() { [[ " $sessions " == *" $1 "* ]]; }

mkdir -p "$mdm/$run/eval"
cd "$mdm"
if ! ls $run/model*.pt >/dev/null 2>&1; then   # fresh start: keep the code this run was started with
    mkdir -p $run/code_snapshot && cp "$here"/*.py "$here"/*.sh "$here/../tier0"/*.py $run/code_snapshot/
fi

if ! want train; then :
elif ! tmux has-session -t e003-train 2>/dev/null; then
    # Restart up to 5 times after a crash; each restart resumes from the newest checkpoint.
    tmux new-session -d -s e003-train "cd $mdm && export TQDM_MININTERVAL=30 && \
        for try in 1 2 3 4 5 6; do \
            .venv/bin/python $here/train_instrumented.py $train_args >> $run/train.stdout.log 2>> $run/train.stderr.log && break; \
            echo \"\$(date '+%F %T') training exited with an error (try \$try), see $run/train.stderr.log\" | tee -a $run/restarts.log; \
            sleep 60; \
        done; echo 'training session finished'; exec bash"
    echo "started tmux session e003-train"
else
    echo "e003-train already running"
fi

if ! want eval; then :
elif ! tmux has-session -t e003-eval 2>/dev/null; then
    tmux new-session -d -s e003-eval "cd $mdm && \
        until [ -f $run/args.json ]; do sleep 10; done; \
        .venv/bin/python $here/watch.py --run $run --num_steps $steps 2>&1 | tee -a $run/eval/watch.log; exec bash"
    echo "started tmux session e003-eval"
else
    echo "e003-eval already running"
fi
if ! want tb; then :
elif ! tmux has-session -t e003-tb 2>/dev/null; then
    tmux new-session -d -s e003-tb "cd $mdm && .venv/bin/tensorboard --logdir $run --port 6006; exec bash"
    echo "started tmux session e003-tb (http://localhost:6006)"
else
    echo "e003-tb already running"
fi
echo "logs: $mdm/$run/{train.stdout.log,train_log.jsonl,eval/watch.log}"
