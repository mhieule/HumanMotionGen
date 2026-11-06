#!/usr/bin/env bash
# Side-by-side text-to-motion: MDM vs. SALAD vs. MotionHiFlow on the same prompts.
#
#   scripts/compare/compare.sh <run name> <prompts.txt> [frames=120] [reps=3] [seed=10]
#
# prompts.txt: one prompt per line. Every model gets the same length (frames at
# 20 fps; divisible by 4, max 196) and the same number of repetitions.
# Pick models with MODELS="salad mhf" (default: all three).
# Output: comparisons/<run>/{mdm,salad,mhf}/pKK_rR.npy and comparisons/<run>/grid_pKK.mp4
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
run=${1:?usage: compare.sh <run name> <prompts.txt> [frames] [reps] [seed]}
prompts=$(realpath "${2:?usage: compare.sh <run name> <prompts.txt> [frames] [reps] [seed]}")
frames=${3:-120}; reps=${4:-3}; seed=${5:-10}
models=${MODELS:-mdm salad mhf}
out=$root/comparisons/$run
mkdir -p "$out"
grep -v '^[[:space:]]*$' "$prompts" > "$out/prompts.txt"
here=$root/scripts/compare

for m in $models; do
    case $m in
    salad)
        # PyTorch 1.13's cuDNN 8 dlopens "libcuda.so"; on WSL that lives in /usr/lib/wsl/lib.
        (cd "$root/baselines/salad" && LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} .venv/bin/python "$here/gen_salad.py" \
            --prompts "$out/prompts.txt" --out "$out/salad" --frames "$frames" --reps "$reps" --seed "$seed") ;;
    mhf)
        (cd "$root/baselines/MotionHiFlow" && .venv/bin/python "$here/gen_mhf.py" \
            --prompts "$out/prompts.txt" --out "$out/mhf" --frames "$frames" --reps "$reps" --seed "$seed") ;;
    mdm)
        command -v docker >/dev/null || { echo "docker not found in WSL: enable Docker Desktop > Settings > Resources > WSL integration for this distro"; exit 1; }
        # The container only sees mdm_assets/save, so stage the prompts there.
        stage=$root/mdm_assets/save/compare_$run
        rm -rf "$stage" && mkdir -p "$stage" && cp "$out/prompts.txt" "$stage/"
        secs=$(python3 -c "print($frames / 20)")
        HOST_UID=$(id -u) HOST_GID=$(id -g) DOCKER_CONFIG=$root/docker/client-config \
        docker compose -f "$root/docker/compose.yaml" run --rm -e PYTHONPATH=/mdm mdm bash -c \
            "CKPT=\$(ls save/*50steps*/model*.pt | head -n 1); python -m sample.generate --model_path \$CKPT \
             --input_text save/compare_$run/prompts.txt --motion_length $secs --num_repetitions $reps \
             --seed $seed --output_dir save/compare_$run/out \
             && python scripts/compare/mdm_to_joints.py save/compare_$run/out save/compare_$run/joints"
        rm -rf "$out/mdm" && cp -r "$stage/joints" "$out/mdm" ;;
    *) echo "unknown model: $m"; exit 1 ;;
    esac
done

"$root/baselines/MotionHiFlow/.venv/bin/python" "$here/render_grid.py" "$out"
echo "Done: $out"
