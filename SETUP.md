# Set up MDM on a new machine and train it

From a fresh Linux machine to a running MDM training on HumanML3D. No Docker needed. Works on RTX 50xx (Blackwell, e.g. the lab PC's RTX 5070 Ti) and on older cards.

> For the Docker setup on the RTX 3060 laptop (sampling, evaluation), see [MDM_HumanML3D_RTX3060.md](MDM_HumanML3D_RTX3060.md). That image has PyTorch 1.13 and **does not run on RTX 50xx** ("no kernel image is available").

## What you need

| | |
|---|---|
| OS | Linux (Ubuntu 22.04 / 24.04), `sudo` for `apt` |
| GPU | NVIDIA, ≥ 12 GB VRAM (default training uses ~11 GB) |
| Driver | **≥ 570** for RTX 50xx (`nvidia-smi` shows the version). The CUDA toolkit is *not* needed: PyTorch brings its own. |
| Disk | ~40 GB free (dataset 9.5 GB unpacked, data caches 8.3 GB, Python env 5.5 GB, evaluators/models 1.5 GB, archives while unpacking, ~0.3 GB per training checkpoint) |
| Download | ~8 GB in total (dataset ~5 GB, PyTorch + libraries ~3 GB, assets ~1.5 GB) |

> **Tested (2026-11-06)** on the RTX 3060 (WSL2 Ubuntu 22.04, driver 595): all setup steps, sampling, and a 300-step training with in-training evaluation and sample rendering. That test used PyTorch 2.6.0 + CUDA 12.6 (already cached locally); the script installs 2.7.1 + CUDA 12.8 by default, which has not been run on an RTX 50xx yet. The `dataset` step was tested with existing files; the fresh Drive download was not run end to end.

## 1. Clone this repo

```bash
cd /workspace
git clone git@github.com:mhieule/HumanMotionGen.git
cd HumanMotionGen
```

If the repo is private and the machine has no GitHub key yet: `ssh-keygen -t ed25519`, add `~/.ssh/id_ed25519.pub` under GitHub → Settings → SSH keys, then clone. (Or clone over HTTPS with a personal access token.)

## 2. Run the setup script

```bash
scripts/setup_mdm.sh
```

It is safe to re-run: every step skips what is already there. Run it inside `tmux` if your SSH connection might drop (`tmux new -s setup`, detach with `Ctrl-b d`, reattach with `tmux attach -t setup`).

What it does, in order (`STEPS="..."` runs a subset, e.g. `STEPS="dataset check" scripts/setup_mdm.sh`):

| Step | What | Where |
|---|---|---|
| `tools` | checks `nvidia-smi`; installs `git curl unzip ffmpeg` and a .rar extractor (`unrar`, or `bsdtar` as fallback) via apt if missing; installs [uv](https://docs.astral.sh/uv/) | system, `~/.local/bin` |
| `repo` | clones [MDM](https://github.com/GuyTevet/motion-diffusion-model) at commit `ef8edce` | `baselines/mdm/` |
| `venv` | Python 3.11 env with **PyTorch 2.7.1 + CUDA 12.8** (first build with Blackwell kernels) and MDM's dependencies; downloads CLIP ViT-B/32 | `baselines/mdm/.venv`, `~/.cache/clip` |
| `assets` | SMPL body model, GloVe, Guo et al. T2M evaluators, pretrained MDM-50 checkpoint | `mdm_assets/` |
| `dataset` | processed HumanML3D from Google Drive (+ captions from the HumanML3D GitHub repo), unpacked | `mdm_assets/dataset/HumanML3D/` |
| `links` | symlinks MDM's expected folders to `mdm_assets/` | `baselines/mdm/{dataset/HumanML3D, body_models/smpl, glove, t2m, save}` |
| `check` | GPU + PyTorch arch check, imports, file counts | |

Everything it creates is gitignored (`baselines/`, `mdm_assets/`).

**Expected `check` output** (the GPU line differs):

```
torch 2.7.1+cu128 | CUDA 12.8 | GPU NVIDIA GeForce RTX 5070 Ti
GPU arch sm_120 supported by this build: True
matmul ok: True
clip / smplx / chumpy / spacy ok
dataset/HumanML3D/new_joint_vecs     29228 files (expect ~29.2k)
dataset/HumanML3D/new_joints         29228 files (expect ~29.2k)
dataset/HumanML3D/texts              29232 files (expect ~29.2k)
ok       dataset/HumanML3D/Mean.npy
...
ok       save/humanml_enc_512_50steps/model000750000.pt
```

`supported by this build: False` means the PyTorch build has no kernels for this GPU, see Troubleshooting.

## 3. Smoke test: sample from the pretrained model (~1 min)

All MDM commands run from `baselines/mdm` with its venv:

```bash
cd /workspace/HumanMotionGen/baselines/mdm
source .venv/bin/activate
python -m sample.generate --model_path save/humanml_enc_512_50steps/model000750000.pt \
    --text_prompt "a person jumps forward" --num_repetitions 2 --output_dir save/smoke_sample
```

Output: `save/smoke_sample/*.mp4` (stick figures) and `results.npy`. `save/` is `mdm_assets/save/` in the repo root.

## 4. First training run

### 4.1 Short test (~15–20 min)

Catches problems before you start a multi-day run. 1000 steps; MDM evaluates (and renders samples) at step 0, at step 1000 and when it stops at the end of that epoch (step 1149), ~3 min each with `--eval_rep_times 1`:

```bash
python -m train.train_mdm --save_dir save/test_run --overwrite \
    --dataset humanml --diffusion_steps 50 --mask_frames --use_ema \
    --num_steps 1000 --save_interval 1000 --log_interval 100 \
    --eval_during_training --eval_rep_times 1 --gen_during_training \
    --train_platform_type TensorboardPlatform
```

The first start spends ~5 min turning the training split into a cache (`dataset/t2m_train.npy`, 7 GB, stored in `mdm_assets/save/`). Later starts load it in under a minute. Watch the speed in the progress bar: that tells you how long the full run takes (see 4.3).

### 4.2 Full run: reproduce MDM-50

The README's command for the 50-step model (the one we evaluated, FID 0.53 vs. 0.49 in the authors' log), plus in-training evaluation and TensorBoard:

```bash
tmux new -s train
cd /workspace/HumanMotionGen/baselines/mdm && source .venv/bin/activate
python -m train.train_mdm --save_dir save/my_humanml_enc_512_50steps --overwrite \
    --dataset humanml --diffusion_steps 50 --mask_frames --use_ema \
    --eval_during_training --gen_during_training \
    --train_platform_type TensorboardPlatform 2>&1 | tee -a save/my_humanml_enc_512_50steps.log
# detach: Ctrl-b d      reattach: tmux attach -t train
```

| Flag | Meaning |
|---|---|
| `--overwrite` | needed with TensorBoard: MDM creates the log folder before it checks that `save_dir` is new. It does not delete anything. |
| `--diffusion_steps 50` | 50 noise steps instead of 1000: 20× faster sampling and evaluation, same quality |
| `--mask_frames` | fixes a bug where padding frames were attended to |
| `--use_ema` | keeps an exponential moving average of the weights; checkpoints are evaluated with it |
| `--eval_during_training` | at every checkpoint: FID / R-precision / diversity on 1000 test prompts × 3 (~10 min) |
| `--gen_during_training` | at every checkpoint: a few sample videos |
| defaults | batch 64, lr 1e-4, **600k steps**, checkpoint every 50k steps. The released checkpoint is from step 750k; add `--num_steps 750000` to match it. |

Outputs in `save/my_humanml_enc_512_50steps/`: `modelNNNNNNNNN.pt` + `optNNNNNNNNN.pt` per checkpoint (~0.3 GB together), `eval_humanml_NNNNNNNNN.log` per checkpoint (also at step 0, as an untrained baseline), `*.samples/` videos, `args.json`. TensorBoard logs are written there too:

```bash
tensorboard --logdir save/my_humanml_enc_512_50steps --port 6006
# on your laptop: ssh -L 6006:localhost:6006 <lab-pc>, then open http://localhost:6006
```

**Resume** after a crash or reboot: run the exact same command again. MDM finds the newest `model*.pt` in `save_dir` and continues from it (weights, optimizer, step count); at most the steps since the last checkpoint (≤ 50k, a few hours) are lost. For more frequent checkpoints add e.g. `--save_interval 25000` (each eval then costs another ~10 min).

### 4.3 How long it takes

| GPU | Speed | 600k steps | 750k steps | In-training eval (12 checkpoints) |
|---|---|---|---|---|
| RTX 3060 12 GB | 2.2 steps/s (measured) | ~3.2 days | ~4 days | ~2 h |
| RTX 5070 Ti 16 GB | ~4.5–5.5 steps/s (estimate) | ~1.5–2 days | ~1.7–2.3 days | ~1 h |

Training is GPU-bound; the evaluation adds only a few percent. Keep the GPU free of other jobs during the run.

## 5. Evaluate a checkpoint

Standard protocol (1000 test prompts × 20 replications, mean ± 95% CI), ~45 min on the 3060:

```bash
python -m eval.eval_humanml --model_path save/my_humanml_enc_512_50steps/model000600000.pt --eval_mode wo_mm
```

The result goes to `eval_humanml_*_wo_mm.log` next to the checkpoint. **Sanity check:** the ground-truth row must read about R-top1 0.511 / Matching 2.97 / FID 0.002. If it reads R-top1 ~0.45, the epoch-11 evaluator is in use (see below).

Reference numbers (pretrained MDM-50, standard evaluator): R-top3 0.749 · FID 0.49 (authors' log) / 0.53 (our run) · Matching 3.33 · Diversity 9.9.

## How this differs from MDM's own instructions

- **Environment:** MDM's `environment.yml` is Python 3.7 + PyTorch 1.7.1 / CUDA 11.0, which has no kernels for RTX 30xx and newer. `setup_mdm.sh` uses Python 3.11 + PyTorch 2.7.1 / CUDA 12.8 with pinned older libraries where MDM's code needs them (numpy 1.23, matplotlib 3.6, moviepy 1.0.3, setuptools < 70) and patches one line in `chumpy` for Python 3.11.
- **Evaluator:** MDM's `prepare/download_t2m_evaluators.sh` currently ships an epoch-11 `text_mot_match` model. Its numbers are not comparable with papers. The script swaps in the standard epoch-28 one (same as MoMask / MotionHiFlow use); the original stays in `mdm_assets/t2m/text_mot_match_ep11_from_mdm_zip`.
- **Dataset:** processed HumanML3D from Drive folder `1OZrTlAGRvLjXhXwnRiOC-oxYry1vf-Uu` instead of rebuilding it from AMASS. `animations.rar` (preview videos) is skipped.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA error: no kernel image is available` / `sm_120 ... not compatible` | PyTorch build too old for the GPU. Re-run `STEPS=venv scripts/setup_mdm.sh` (default is 2.7.1 + cu128). For a newer version: `TORCH="torch==2.8.0 torchvision==0.23.0" STEPS=venv scripts/setup_mdm.sh`. |
| `torch.cuda.is_available()` is False | Driver missing or too old: `nvidia-smi` must work and show ≥ 570 for RTX 50xx. |
| gdown: "Too many users have viewed or downloaded this file" | Google Drive quota. Wait a few hours and re-run `STEPS="assets dataset" scripts/setup_mdm.sh`; partial downloads resume. |
| `ModuleNotFoundError: pkg_resources` | setuptools too new for wandb 0.15: `uv pip install --python baselines/mdm/.venv "setuptools<70"`. |
| `AttributeError: module 'inspect' has no attribute 'getargspec'` | chumpy patch missing: re-run `STEPS=venv scripts/setup_mdm.sh`. |
| `DataLoader worker ... exited unexpectedly` | Too little shared memory, only inside containers (Docker's default `/dev/shm` is 64 MB; check with `df -h /dev/shm`). Restart the container with `--shm-size 8g`. |
| CUDA out of memory | Another process uses the GPU (`nvidia-smi`). The default config needs ~11 GB. |
| `PermissionError: dataset/t2m_train.npy` | The cache symlink target is not writable: check `mdm_assets/save/` belongs to you. |
