#!/usr/bin/env bash
# Wrapper for the dockerized MDM setup. Run ./mdm.sh with no arguments for help.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export HOST_UID=$(id -u) HOST_GID=$(id -g)
# Project-local client config: ~/.docker uses desktop.exe for credentials,
# which WSL cannot run when Windows interop is off. Public images need none.
export DOCKER_CONFIG="$here/docker/client-config"
dc() { docker compose -f "$here/docker/compose.yaml" "$@"; }

# Bind-mount sources must exist before the container starts, owned by you.
mkdir -p "$here"/mdm_assets/{dataset/HumanML3D,body_models/smpl,glove,save,gt_vis}

CKPT='$(ls save/*50steps*/model*.pt | head -n 1)'
run() { dc run --rm -e PYTHONPATH=/mdm mdm bash -c "CKPT=$CKPT; $1"; }

case "${1:-}" in
    build)  dc build ;;
    fetch)  dc run --rm mdm bash scripts/fetch_assets.sh ;;
    check)
        run 'python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
             for p in new_joint_vecs new_joints texts test.txt Mean.npy Std.npy; do
                 [ -e dataset/HumanML3D/$p ] && echo "ok       dataset/HumanML3D/$p" || echo "MISSING  dataset/HumanML3D/$p"
             done
             ls $CKPT body_models/smpl/SMPL_NEUTRAL.pkl glove/our_vab_data.npy' ;;
    peek)   # Step 2: one test motion
        run 'python -c "
import numpy as np
r=\"dataset/HumanML3D\"; mid=open(f\"{r}/test.txt\").readline().strip()
print(mid, np.load(f\"{r}/new_joint_vecs/{mid}.npy\").shape, np.load(f\"{r}/new_joints/{mid}.npy\").shape)
print(open(f\"{r}/texts/{mid}.txt\").read())"' ;;
    random) # Step 4.1: random test-set prompts at ground-truth lengths
        run 'python -m sample.generate --model_path $CKPT --num_samples 6 --num_repetitions 3' ;;
    prompt) # Step 4.2: ./mdm.sh prompt "a person ..." [seconds]
        text=${2:?usage: ./mdm.sh prompt \"text\" [seconds]}
        run "python -m sample.generate --model_path \$CKPT --text_prompt $(printf %q "$text") --motion_length ${3:-6} --num_repetitions 3" ;;
    compare) # Step 4.3: real test motions, then MDM on the same captions
        run 'python scripts/compare_gt.py &&
             python -m sample.generate --model_path $CKPT --input_text gt_vis/prompts.txt --motion_length 6 --num_repetitions 3' ;;
    stats)  # ./mdm.sh stats <samples folder name or path> [--gt-words walk circle]
        dir=save/humanml_enc_512_50steps/$(basename "${2:?usage: ./mdm.sh stats <samples folder>}")
        run "python scripts/motion_stats.py $(printf %q "$dir") ${*:3}" ;;
    shell)  dc run --rm -e PYTHONPATH=/mdm mdm bash ;;
    clean)  # Remove the image and build cache; keep mdm_assets unless --all
        dc down --rmi all --remove-orphans || true
        docker builder prune -f
        if [ "${2:-}" = "--all" ]; then rm -rf "$here/mdm_assets"; echo "Removed mdm_assets/"; fi ;;
    *) cat <<'EOF'
Usage: ./mdm.sh <command>
  build              build the image (once, roughly 6-8 GB, 10-20 min)
  fetch              download SMPL, GloVe and the 50-step model into mdm_assets/
  check              GPU + dataset + asset sanity check
  peek               Step 2: shapes and captions of one test motion
  random             Step 4.1: 6 random test prompts x 3 repetitions
  prompt "text" [s]  Step 4.2: your own prompt (default 6 s, max 9.8)
  compare            Step 4.3: real vs. generated for 5 test captions
  stats <folder>     path / foot-sliding / floating / jitter stats vs. real clips
  shell              interactive shell in the container (CKPT not set; see docs)
  clean [--all]      remove the image and build cache (--all also deletes mdm_assets/)
Outputs: mdm_assets/save/<model>/samples_*/ and mdm_assets/gt_vis/
EOF
    ;;
esac
