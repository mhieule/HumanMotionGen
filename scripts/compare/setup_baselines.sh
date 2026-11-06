#!/usr/bin/env bash
# Recreate baselines/ (gitignored): SALAD and MotionHiFlow, each with its own uv venv,
# pretrained HumanML3D checkpoints and the files needed for generation. Needs uv,
# git, unzip. Re-running skips what is already there.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$root/baselines" && cd "$root/baselines"
export UV_HTTP_TIMEOUT=300

clone() {  # url dir commit
    [ -d "$2" ] || git clone -q "$1" "$2"
    git -C "$2" checkout -q "$3"
}

# --- SALAD (CVPR 2025): Python 3.9, PyTorch 1.13.1 + cu117 ---
clone https://github.com/seokhyeonhong/salad.git salad 6487e98
(
    cd salad
    [ -d .venv ] || uv venv -q -p 3.9 .venv
    export VIRTUAL_ENV=$PWD/.venv
    uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117 --index-strategy unsafe-best-match
    # requirements.txt leaves transformers unpinned; 4.39.3 matches its tokenizers==0.15.2
    sed 's/^transformers$/transformers==4.39.3/' requirements.txt > .req-pinned.txt
    uv pip install -r .req-pinned.txt
    # Pretrained HumanML3D VAE + denoiser + evaluator meta (prepare/download_t2m.sh)
    [ -d checkpoints/t2m/t2m_denoiser_vpred_vaegelu ] || \
        (mkdir -p checkpoints && cd checkpoints && uvx gdown --folder "https://drive.google.com/drive/folders/1YuDQCgc6RJ4WlR9vt_L34nkXChC7C98_")
    # Training/eval read ./dataset/humanml3d and ./glove: reuse MDM's copies
    ln -sfn ../../../mdm_assets/dataset/HumanML3D dataset/humanml3d
    ln -sfn ../../mdm_assets/glove glove
)

# --- MotionHiFlow (CVPR 2026): Python 3.11, PyTorch 2.6 + cu126 ---
clone https://github.com/ai-lh/MotionHiFlow.git MotionHiFlow e0a1605
(
    cd MotionHiFlow
    [ -d .venv ] || uv venv -q -p 3.11 .venv
    export VIRTUAL_ENV=$PWD/.venv
    uv pip install -r requirements.txt --index-strategy unsafe-best-match
    # HumanML3D (t2m) evaluator: generation also reads its mean/std (configs/resources.yaml)
    if [ ! -f deps/evaluators/t2m/Comp_v6_KLD005/meta/mean.npy ]; then
        mkdir -p deps/evaluators/t2m
        uvx gdown 19C_eiEr0kMGlYVJy_yFL6_Dhk3RvmwhM -O deps/evaluators/t2m/ev.zip
        unzip -q -o deps/evaluators/t2m/ev.zip -d deps/evaluators/t2m && rm deps/evaluators/t2m/ev.zip
    fi
    # Pretrained HumanML3D VAE + flow model, HumanML3D only (prepare.sh's fallback source; skips KIT)
    if [ ! -f logs/t2m_tmdit_16d/checkpoints/net_best_fid.tar ]; then
        tmp=$(mktemp -d)
        uvx --from "huggingface_hub[cli]" hf download heng-li/MotionHiFlow --include "checkpoints/t2m_*" --local-dir "$tmp"
        mkdir -p logs && cp -a "$tmp"/checkpoints/t2m_* logs/ && rm -rf "$tmp"
    fi
)
echo "Baselines ready in $root/baselines"
