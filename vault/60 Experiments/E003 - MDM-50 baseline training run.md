---
tags: [experiment]
id: E003
date: 2026-11-14
status: running
model: MDM
dataset: HumanML3D
representation: 263-D
target: x0
baseline: "released MDM-50 checkpoint (E001)"
commit: "e7bac0a + uncommitted experiments/baseline/ (copy in the run's code_snapshot/)"
config: "experiments/baseline/run_e003.sh"
gpu_hours: 
fid: 
r_precision_top3: 
---
# E003: MDM-50 baseline training run (learning curves, proxy length)

## Question / hypothesis
1. Does our pipeline reproduce the released MDM-50 checkpoint when we train it ourselves?
2. How do the metrics grow with training steps and hours, and how short can a proxy run be? ([[Meeting 03]], "proxy check first")

## Setup
- **Recipe = the released checkpoint's `args.json`**: `--dataset humanml --diffusion_steps 50`, seed 10, batch 64, lr 1e-4, **no `--mask_frames`**, 750k steps (the released checkpoint is from step 750k). `--use_ema` only keeps an averaged copy; the raw weights are the reproduction.
- Command: `experiments/baseline/run_e003.sh` (tmux `e003-train`, `e003-eval`, `e003-tb`). Run directory: `mdm_assets/save/E003_mdm50_baseline/`.
- Logging: every 100 steps to TensorBoard (`http://localhost:6006`) and `train_log.jsonl`; checkpoint every 5k steps, all kept.
- Evaluation in a separate process, **val** split, raw and EMA weights: benchmark (1000 motions × 3 reps, CFG 2.5) every 5k to 50k, 10k to 200k, 25k to 750k; held-out denoising loss; APG scale 4 every 50k; **test** (20 reps) at 600k and 750k only.
- **Proxy-length criterion:** FID, R-top3 and MM-Dist all within **5%** / **10%** of their final values, from that checkpoint on (`analyze.py`).
- Hardware: lab PC, RTX 5070 Ti, ~5.4 steps/s alone, 4 GB VRAM. Expected ~2.2 days including evaluation.
- Design, all decisions and deviations from the released recipe: `docs/superpowers/specs/2026-11-14-mdm-baseline-run-design.md`. Harness: `experiments/baseline/README.md`.

## Results
Generated: `experiments/baseline/RESULTS.md` (re-run `analyze.py` any time).

| Run | FID ↓ | R-Prec top-3 ↑ | MM-Dist ↓ | Diversity → | Notes |
|---|---|---|---|---|---|
| released checkpoint, test, 20 reps | 0.552 ± 0.047 | 0.751 ± 0.004 | 3.333 | 9.86 | with the kernel fix (`save/tier0_fixed/`); E002 measured 0.567 ± 0.049 with the faulty kernel |
| ours, 600k, test | | | | | |
| ours, 750k, test | | | | | |

## Samples
<!-- Standard visual set (8 prompts × 3 seeds): not generated yet; all checkpoints are kept -->

## Observations
- **PyTorch's memory-efficient attention kernel is wrong on the lab PC** (RTX 5070 Ti, PyTorch 2.7.1 + CUDA 12.8) for float32 with head size > 64; MDM's is 128 → [[Attention kernel bug on the lab PC]]. The first launch diverged at step ~600 (gradient norm 0.4 → 2000+, loss 0.35 → 1.15); so did six probes with three seeds, with and without `--mask_frames`. Replaying the spike step: gradient norm ~4000–7700 on GPU vs. 0.32 on CPU with identical weights and inputs. First deviating module: layer 0 self-attention. Fix: `torch.backends.cuda.enable_mem_efficient_sdp(False)` (now in all our scripts; `scripts/fixed_attention.py -m ...` for stock MDM commands; `scripts/check_attention_kernel.py` tests an environment).
- With the fix, training is clean (3000-step probes: loss 0.52 → 0.18, no spikes) and **bit-reproducible**: two stock MDM runs, our launcher, and our launcher with EMA give identical weights after 200 steps.
- **Earlier lab-PC numbers used the faulty kernel for sampling** ([[E002 - Inference-time guidance on MDM (Tier 0)]], and the lab-PC FID 0.567 of the released checkpoint). Per denoising step the error is 0.005 on average with ~0.1% of values off by > 0.1. Re-evaluated with the fix (same seeds): test FID 0.567 → 0.552 (CFG 2.5) and 0.463 → 0.444 (APG scale 4), everything else unchanged to the second decimal. Inside the CIs; the E002 conclusions hold.
- MDM's loop counts epochs: one epoch is 383 steps (24,546 training items), and after a resume it would run the full epoch count again. The launcher stops at exactly `num_steps`.

## Conclusion / next
