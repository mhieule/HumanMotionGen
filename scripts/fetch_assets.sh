#!/usr/bin/env bash
# Runs inside the container (via ./mdm.sh fetch). Downloads the small assets
# MDM needs for sampling into the mounted folders; skips what is already there.
set -euo pipefail
cd /mdm
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fetch_zip() {  # name, gdown id, check path, unzip-into dir
    local name=$1 id=$2 check=$3 dest=$4
    if [ -e "$check" ]; then echo "[skip] $name already present"; return; fi
    echo "[get]  $name"
    gdown --fuzzy "https://drive.google.com/file/d/$id/view" -O "$tmp/$name.zip"
    unzip -q -o "$tmp/$name.zip" -d "$tmp/$name"
    # Each zip holds a single top-level folder; move its contents into the mount.
    local inner
    inner=$(find "$tmp/$name" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    cp -r "$inner"/. "$dest"/
}

fetch_zip smpl  1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2 body_models/smpl/SMPL_NEUTRAL.pkl body_models/smpl
fetch_zip glove 1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n glove/our_vab_data.npy             glove

# The checkpoint zip keeps its folder (save/humanml_enc_512_50steps/...).
if ls save/*50steps*/model*.pt >/dev/null 2>&1; then
    echo "[skip] 50-step model already present"
else
    echo "[get]  humanml-encoder-512-50steps"
    gdown --fuzzy "https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view" -O "$tmp/model.zip"
    unzip -q -o "$tmp/model.zip" -d save
fi

echo
ls save/*50steps*/model*.pt
