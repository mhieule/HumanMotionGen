---
tags: [experiment]
id: E001
date: 2026-11-05
status: in progress
model: MDM
dataset: HumanML3D
representation: 263-D
target: x0
baseline: "paper numbers"
commit: 
config: 
gpu_hours: 
fid: 0.527
r_precision_top3: 0.749
---
# E001: Reproduce MDM on HumanML3D

## Question / hypothesis
Can we train MDM on the 10–12 GB lab PC and get close to the paper's numbers? This is the baseline for every later ablation (slide milestone 1: "train an existing baseline from GitHub").

## Setup
- Repo: [[MDM]] · Data: [[HumanML3D]]
- Changes from the default config: (batch size? steps?)
- Hardware: lab PC, GPU = ?, VRAM = ?

## Steps
- [x] Evaluate the **pretrained** checkpoint first → checks the evaluator and data pipeline (2026-11-05, matches the authors' log)
- [ ] Short training run (e.g. 10k steps) → loss curve + samples, VRAM and speed
- [ ] Full run (over a weekend?), with checkpoints evaluated along the way

## Results
| Checkpoint | FID ↓ | R-Prec top-3 ↑ | MM-Dist ↓ | Diversity → | Notes |
|---|---|---|---|---|---|
| Real motions (GT row) | 0.002 | 0.798 | 2.970 | 9.458 | target values; identical in both logs |
| Authors' log (shipped with ckpt) | 0.489 ± 0.047 | 0.749 ± 0.006 | 3.326 ± 0.025 | 9.924 ± 0.083 | `authors_eval_wo_mm_2024-04-15.log` |
| Pretrained (our eval) | 0.527 ± 0.047 | 0.749 ± 0.006 | 3.328 ± 0.025 | 9.949 ± 0.090 | `wo_mm`, 20 reps, RTX 3060, ~45 min; R-top1 0.453 |
| Ours | | | | | |

## Samples
<!-- Embed videos: same prompts and seeds as the standard visual set; denoising trajectory at t = T, ¾T, ½T, ¼T, 0 -->

## Observations
- **Use the epoch-28 evaluator.** MDM's current `prepare/download_t2m_evaluators.sh` fetches a different `text_mot_match/model/finest.tar` (epoch 11). With it, the ground-truth row drops to R-top1 0.454 and Matching 3.24, so its numbers aren't comparable with papers. Swapped in the standard epoch-28 one (identical to MotionHiFlow's `deps/evaluators/t2m`). The epoch-11 copy is kept as `mdm_assets/t2m/text_mot_match_ep11_from_mdm_zip`.
- **Sanity check for every baseline:** the ground-truth row must read about R-top1 0.511 / Matching 2.97 / FID 0.002.
- Docker fixes needed for the eval: `wandb` import, writable `dataset/t2m_test.npy` cache (symlinked into `save/`), `shm_size: 8gb`.
- **Training speed (measured 2026-11-05, RTX 3060, MDM-50 config, batch 64):** ~2.2 steps/s, GPU-bound (GPU 70–99%, CPU mostly idle), ~11 GB VRAM. That gives 600k steps (default) ≈ 3.2 days and 750k (released ckpt) ≈ 4 days. In-training eval (3 reps × 1000 samples every 50k steps) adds about 10 min per checkpoint, ~2 h in total, so it's negligible. Training also needed `blobfile` and a writable `t2m_train.npy` cache (both now in the Dockerfile).
- **Lab PC (RTX 5070 Ti 16 GB), estimate, not measured:** ~2–2.5× the 3060, so ≈ 1.5–2 days for 600k steps. **The current image won't run there:** Blackwell (sm_120) needs PyTorch ≥ 2.7 with CUDA 12.8, while `mdm-3060` is torch 1.13 + cu117 on Python 3.8. It needs a new image (Python 3.10, torch 2.7+cu128). Check the eval reproduces first, then time ~1k steps.
- **SALAD for comparison (measured on the 3060, official defaults, both stages GPU-bound):**
  - VAE: batch 256, 50 epochs = 361k iterations at ~0.42 s each ≈ 42 h.
  - Denoiser: batch 64, 500 epochs = 191k iterations at ~0.55 s each ≈ 29 h, ~12 GB VRAM (at the 3060's limit). Startup takes ~5 min to load data.
  - From scratch ≈ 3 days, about the same as MDM. The denoiser alone on the released VAE ≈ 1.2 days.
- Command: `docker compose -f docker/compose.yaml run --rm -e PYTHONPATH=/mdm mdm bash -c 'python -m eval.eval_humanml --model_path save/humanml_enc_512_50steps/model000750000.pt --eval_mode wo_mm'`. Each replication takes ~2 min on the 3060.

## Conclusion / next
