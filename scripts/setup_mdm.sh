#!/usr/bin/env bash
# Native (no Docker) MDM setup on a fresh Linux machine with an NVIDIA GPU, incl. RTX 50xx (Blackwell).
#
#   scripts/setup_mdm.sh                      # all steps; safe to re-run, skips what is done
#   STEPS="dataset links" scripts/setup_mdm.sh   # only some steps
#
# Steps: tools repo venv assets dataset links check
# Layout (all gitignored):
#   baselines/mdm/          MDM code (GuyTevet/motion-diffusion-model @ MDM_COMMIT) + .venv
#   mdm_assets/             dataset, SMPL, GloVe, evaluators, checkpoints, training runs
# baselines/mdm/{dataset/HumanML3D,body_models/smpl,glove,t2m,save} are symlinks into mdm_assets/.
#
# Overrides: PY=3.11  TORCH="torch==2.7.1 torchvision==0.22.1"  TORCH_INDEX=https://download.pytorch.org/whl/cu128
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
mdm=$root/baselines/mdm
assets=$root/mdm_assets
MDM_COMMIT=ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778
PY=${PY:-3.11}
# PyTorch >= 2.7 with CUDA 12.8 is the first build with Blackwell (sm_120) kernels; it also runs on RTX 20/30/40.
TORCH=${TORCH:-"torch==2.7.1 torchvision==0.22.1"}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
steps=${STEPS:-tools repo venv assets dataset links check}
export UV_HTTP_TIMEOUT=300

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
has() { [[ " $steps " == *" $1 "* ]]; }
gd() { "$mdm/.venv/bin/gdown" --continue "$@"; }  # Google Drive download by file id

# ---------------------------------------------------------------- tools
if has tools; then
    say "tools"
    command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first (>= 570 for RTX 50xx)"; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    SUDO=$([ "$(id -u)" = 0 ] || echo sudo)  # root inside containers has no sudo
    need=()
    for t in git curl unzip ffmpeg; do command -v $t >/dev/null || need+=("$t"); done
    if ((${#need[@]})); then
        echo "installing: ${need[*]}"
        $SUDO apt-get update -qq && $SUDO apt-get install -y -qq "${need[@]}"
    fi
    # .rar extractor: unrar is in Ubuntu's multiverse; bsdtar (libarchive-tools, universe) also reads rar
    if ! command -v unrar >/dev/null && ! command -v bsdtar >/dev/null; then
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq unrar || $SUDO apt-get install -y -qq libarchive-tools
    fi
    if ! command -v uv >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH=$HOME/.local/bin:$PATH
    fi
    uv --version
fi
export PATH=$HOME/.local/bin:$PATH

# ---------------------------------------------------------------- repo
if has repo; then
    say "MDM repo -> baselines/mdm @ ${MDM_COMMIT:0:7}"
    [ -d "$mdm/.git" ] || git clone -q https://github.com/GuyTevet/motion-diffusion-model.git "$mdm"
    git -C "$mdm" checkout -q "$MDM_COMMIT"
fi

# ---------------------------------------------------------------- venv
if has venv; then
    say "Python $PY venv + PyTorch ($TORCH from $TORCH_INDEX)"
    cd "$mdm"
    [ -d .venv ] || uv venv -q -p "$PY" .venv
    export VIRTUAL_ENV=$mdm/.venv
    uv pip install -q $TORCH --index-url "$TORCH_INDEX"
    # numpy < 1.24: MDM and chumpy still use np.float / np.int / np.bool.
    # matplotlib < 3.7: plot_script.py calls ax.grid(b=False).  moviepy 1.x: code imports moviepy.editor.
    # setuptools < 70: wandb 0.15 (imported by eval/train) needs pkg_resources.
    uv pip install -q "numpy==1.23.5" "scipy==1.10.1" "matplotlib==3.6.3" "scikit-learn==1.2.2" \
        h5py tqdm trimesh joblib "smplx==0.1.28" ftfy regex "moviepy==1.0.3" imageio \
        "gdown>=5.2" "wandb==0.15.12" "blobfile==2.1.1" tensorboard "spacy==3.7.5" \
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl" \
        "clip @ git+https://github.com/openai/CLIP.git" "setuptools<70" wheel pip
    # chumpy (needed to unpickle SMPL_NEUTRAL.pkl) imports numpy in setup.py
    uv pip install -q --no-build-isolation "chumpy==0.70"
    # chumpy 0.70 calls inspect.getargspec, removed in Python 3.11; getfullargspec has the same .args
    sed -i 's/inspect\.getargspec(/inspect.getfullargspec(/' .venv/lib/python*/site-packages/chumpy/ch.py
    # Pre-fetch the CLIP ViT-B/32 weights (~350 MB) into ~/.cache/clip (quiet: no progress bar in logs)
    .venv/bin/python -c "import clip; clip.load('ViT-B/32', device='cpu')" 2>&1 | tail -n 1
fi

# ---------------------------------------------------------------- assets
fetch_zip() {  # name, gdrive id, check path, dest dir: zip holds one top-level folder whose contents go to dest
    local name=$1 id=$2 check=$3 dest=$4 tmp
    if [ -e "$check" ]; then echo "[skip] $name"; return; fi
    echo "[get]  $name"
    tmp=$(mktemp -d); gd "$id" -O "$tmp/$name.zip"
    unzip -q -o "$tmp/$name.zip" -d "$tmp/x"
    mkdir -p "$dest"; cp -r "$(find "$tmp/x" -mindepth 1 -maxdepth 1 -type d | head -n 1)"/. "$dest"/
    rm -rf "$tmp"
}
if has assets; then
    say "assets -> mdm_assets/ (SMPL, GloVe, T2M evaluators, pretrained MDM-50)"
    mkdir -p "$assets"/{dataset/HumanML3D,body_models/smpl,glove,save,t2m}
    fetch_zip smpl  1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2 "$assets/body_models/smpl/SMPL_NEUTRAL.pkl" "$assets/body_models/smpl"
    fetch_zip glove 1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n "$assets/glove/our_vab_data.npy" "$assets/glove"
    # Guo et al. evaluators from MDM's prepare/download_t2m_evaluators.sh ...
    fetch_zip t2m   1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb "$assets/t2m/Comp_v6_KLD01/meta/mean.npy" "$assets/t2m"
    # ... but that zip's text_mot_match is epoch 11 (GT R-top1 0.454). Use the standard epoch-28 one
    # (GT R-top1 0.511, what papers report), from the MoMask/MotionHiFlow evaluator zip.
    if [ ! -e "$assets/t2m/.text_mot_match_ep28" ]; then
        echo "[get]  standard text_mot_match (epoch 28)"
        tmp=$(mktemp -d); gd 19C_eiEr0kMGlYVJy_yFL6_Dhk3RvmwhM -O "$tmp/ev.zip"; unzip -q -o "$tmp/ev.zip" -d "$tmp/x"
        src=$(find "$tmp/x" -type d -name text_mot_match | head -n 1)
        [ -d "$assets/t2m/text_mot_match" ] && mv "$assets/t2m/text_mot_match" "$assets/t2m/text_mot_match_ep11_from_mdm_zip"
        cp -r "$src" "$assets/t2m/text_mot_match" && touch "$assets/t2m/.text_mot_match_ep28"; rm -rf "$tmp"
    else echo "[skip] standard text_mot_match"; fi
    if ls "$assets"/save/humanml_enc_512_50steps/model*.pt >/dev/null 2>&1; then echo "[skip] pretrained MDM-50"
    else
        echo "[get]  pretrained MDM-50 (humanml_enc_512_50steps)"
        tmp=$(mktemp -d); gd 1cfadR1eZ116TIdXK7qDX1RugAerEiJXr -O "$tmp/m.zip"; unzip -q -o "$tmp/m.zip" -d "$assets/save"; rm -rf "$tmp"
    fi
fi

# ---------------------------------------------------------------- dataset
if has dataset; then
    say "HumanML3D -> mdm_assets/dataset/HumanML3D"
    d=$assets/dataset/HumanML3D; mkdir -p "$d"; cd "$d"
    # Processed HumanML3D, Drive folder 1OZrTlAGRvLjXhXwnRiOC-oxYry1vf-Uu (animations.rar skipped: preview videos)
    while read -r id f; do
        [ -e "$f" ] || [ -e "${f%.rar}" ] && { echo "[skip] $f"; continue; }
        echo "[get]  $f"; gd "$id" -O "$f"
    done <<'EOF'
14HIdUa_LPGEh-M1ICDZJp0tLJftewnAX new_joint_vecs.rar
1uQncXvbH9fxi_WanKeUZ8QIQUGtHCw_O new_joints.rar
1eu4IBLY6uXEVTzHe5VdHyCyeWu0lKyvr Mean.npy
1Ekc31i6oCPJjhoDcKScGIFfTI1J1Rt4q Std.npy
1UoXcvYSkutLgUXQNn5-OgtfPNpnA_k-f all.txt
10-LBP-VC1N4hnMiZ1cJYdxZQsAk5852y test.txt
1RTtT8p2Xt34pM1L0i0T8Q6M55vS_HFxx train.txt
1IzeaUk04ezbE-mHH-ABXvjcCWLgp0Dhl train_val.txt
1WMnL0RkZyXiCdXI65YfnHFanMM5wAKar val.txt
EOF
    # Captions are not in that folder; they ship with the HumanML3D repo.
    [ -d texts ] || { echo "[get]  texts.zip (GitHub)"; curl -sSL -o texts.zip https://github.com/EricGuo5513/HumanML3D/raw/main/HumanML3D/texts.zip \
                      && unzip -q -o texts.zip && rm texts.zip; }
    for r in new_joint_vecs new_joints; do
        [ -d $r ] && continue
        echo "[unpack] $r.rar"
        if command -v unrar >/dev/null; then unrar x -o+ -idq $r.rar; else bsdtar -xf $r.rar; fi
        rm $r.rar
    done
fi

# ---------------------------------------------------------------- links
if has links; then
    say "symlinks baselines/mdm/* -> mdm_assets/"
    cd "$mdm"
    ln -sfn "$assets/dataset/HumanML3D" dataset/HumanML3D
    ln -sfn "$assets/body_models/smpl" body_models/smpl
    for x in glove t2m save; do ln -sfn "$assets/$x" "$x"; done
    # Train/eval cache each processed split as dataset/t2m_<split>.npy (train ~7 GB); keep them in mdm_assets/save
    for s in train val test; do ln -sfn "$assets/save/.cache_t2m_$s.npy" "dataset/t2m_$s.npy"; done
    ls -l dataset/HumanML3D body_models/smpl glove t2m save | awk '{print $9, $10, $11}'
fi

# ---------------------------------------------------------------- check
if has check; then
    say "check"
    cd "$mdm"
    .venv/bin/python - <<'EOF'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0) if ok else None)
if ok:
    cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
    print("GPU arch", cap, "supported by this build:", cap in torch.cuda.get_arch_list())
    x = torch.randn(1024, 1024, device="cuda"); print("matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
import clip, smplx, chumpy, spacy; spacy.load("en_core_web_sm"); print("clip / smplx / chumpy / spacy ok")
EOF
    for p in dataset/HumanML3D/new_joint_vecs dataset/HumanML3D/new_joints dataset/HumanML3D/texts; do
        printf '%-36s %s files (expect ~29.2k)\n' "$p" "$(ls "$p" 2>/dev/null | wc -l)"
    done
    for p in dataset/HumanML3D/{Mean.npy,Std.npy,train.txt,val.txt,test.txt} body_models/smpl/SMPL_NEUTRAL.pkl \
             glove/our_vab_data.npy t2m/Comp_v6_KLD01/meta/mean.npy t2m/text_mot_match/model/finest.tar; do
        [ -e "$p" ] && echo "ok       $p" || echo "MISSING  $p"
    done
    ls save/humanml_enc_512_50steps/model*.pt 2>/dev/null | sed 's/^/ok       /' || echo "MISSING  save/humanml_enc_512_50steps/model*.pt"
fi
say "done"
