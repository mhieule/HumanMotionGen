# E003: MDM-50 baseline training run, instrumented for proxy-length analysis

Status: **approved and launched** (2026-11-14 19:07) · design approved by Hieu 2026-11-14 · see "What happened at launch" for the attention-kernel finding

## Intent

Train MDM-50 on HumanML3D once, end to end, on the lab PC (RTX 5070 Ti), and record enough along the way to answer two questions:

1. **Does our pipeline reproduce the released checkpoint?** (FID 0.527 ± 0.047 in our evaluation, 0.489 in the authors' log.)
2. **How does quality grow with training steps and wall-clock time, and how short can a proxy run be?** Tier 1 comparisons will use short runs; this run sets their length.

The run's harness (training launcher, checkpoint evaluator, analysis) is reused unchanged by every later run, so later results are comparable with this one.

## Goals, non-goals, anti-goals

- **Goals:** a learning curve of every metric against steps and hours; a recommended proxy length with its evidence; a reproduction verdict against the released checkpoint; a reusable harness.
- **Non-goals:** improving MDM; seed-to-seed noise (needs the 3-seed proxy check from Meeting 03); checking that early rankings predict final rankings (needs a second full configuration).
- **Anti-goals:** a baseline that differs from the released recipe in ways we cannot name; instrumentation that changes what is trained; any use of the test split before the final numbers.

## Principle: the training is the released recipe, everything else only reads

The trained weights must come from the released recipe. Anything we add either runs in a separate process on saved checkpoints, or reads values in the training process without changing the computation or the random stream.

### Audit against the released config

Compared `save/humanml_enc_512_50steps/args.json` with this commit's defaults (`ef8edce`, `--dataset humanml --diffusion_steps 50`): 42 of 49 keys identical, and **every key that affects training matches** (architecture, batch 64, lr 1e-4, no weight decay, no LR anneal, cosine schedule, cond_mask_prob 0.1, seed 10, `mask_frames` off). The 7 differing keys are monitoring switches (`eval_during_training`, `gen_during_training`, `train_platform_type`, `overwrite`) and three keys the current code no longer has (`emb_before_mask`, `keyframe_cond_prob`, `keyframe_cond_type`).

### Remaining deviations

| # | Deviation | Affects trained weights? | Why it is needed |
|---|---|---|---|
| 1 | PyTorch 2.7.1 / CUDA 12.8 / Python 3.11 instead of the authors' environment | Possibly (numerics, library defaults) | Unavoidable: the RTX 5070 Ti has no kernels in older builds. Inference is verified (released checkpoint evaluates within the authors' CI); training is not, and this run is the test. |
| 2 | MDM code at `ef8edce`, not the authors' April 2024 code | Unknown | Unavoidable: the old code is not published. The audit above bounds it to code-level changes, not settings. |
| 3 | `--num_steps 750000` (the released `args.json` says 600000) | Yes, by design | The released checkpoint is `model000750000.pt`. Training to 750k matches the artifact and contains the 600k point. |
| 4 | Checkpoint every 5k steps instead of 50k (`--save_interval 5000`) | No | Needed to locate the proxy length. Saving does not touch the model or the random stream. |
| 5 | Evaluation in a separate process, on **val**, instead of in the training loop on test | No | Val for selection is the Meeting 03 protocol. A separate process leaves the training's random stream untouched and lets us add evaluations later. |
| 6 | `--use_ema`: also keep an averaged copy of the weights | **No** (verified in code, to be confirmed by test) | Optional, see below. |
| 7 | Extra logging in the training process (gradient norm, step time, per-timestep loss) | No (read-only; confirmed by the equivalence test) | "Log aggressively"; MDM logs only the loss and its four timestep quartiles. |
| 8 | PyTorch's memory-efficient attention kernel is switched off (`torch.backends.cuda.enable_mem_efficient_sdp(False)`) | Yes: without it the weights are wrong | That kernel returns wrong values and gradients on the lab PC's GPU; training diverges after ~600 steps. The math kernel used instead is the computation MDM's original PyTorch 1.x performed. See "What happened at launch". |

**Not adopted: `--mask_frames`.** It changes training (padded frames are masked from attention) and the released checkpoint did not use it. MDM's README recommends it, so it becomes its own Tier 1 proxy experiment against this baseline, not part of the baseline. `SETUP.md` section 4.2 currently recommends it for the "reproduce" command and will be corrected.

**On `--use_ema` (deviation 6).** Measured: with the kernel fix, the raw weights after 200 steps are bit-identical with and without the flag. In `train/training_loop.py` the flag does three things: copies the model at start, updates the copy after each optimizer step, and saves it next to the raw weights. The optimizer is AdamW with betas (0.9, 0.999) in both branches. None of these consumes random numbers or writes to the raw weights, so the raw weights are trained as they would be without the flag. Benefit: one run gives both the raw curve (the reproduction) and the EMA curve (needed by Tier 1, where every modern recipe uses EMA, and relevant to proxy length because EMA at 0.9999 lags by about 10k steps). Cost: twice the evaluation time and 82 MB more per checkpoint. **Raw weights are the headline and the reproduction; EMA is a tracked extra.**

## Approaches considered

| Approach | Trade-off | Verdict |
|---|---|---|
| A. MDM's own flags only (`--eval_during_training`, `--save_interval`) | No code. But evaluation is on one fixed schedule, runs inside the training loop (a crash kills training, each evaluation shifts the training's random stream), reports only MDM's four metrics, and cannot evaluate raw and EMA weights side by side. | Rejected |
| **B. Unmodified training recipe + separate checkpoint evaluator** | Small amount of our own code in `experiments/baseline/`. Training saves every 5k steps; a second process evaluates checkpoints on a schedule we can change afterwards. Evaluation shares the GPU, so training slows while it runs. | **Chosen** |
| C. Patch MDM's `training_loop.py` | Most flexible, but edits a third-party checkout that is not in our git, so the baseline could no longer be described as "MDM at `ef8edce`". | Rejected |

## Design

### Training process

`experiments/baseline/train_instrumented.py`, run from `baselines/mdm` in a tmux session `e003-train`. It calls MDM's own data loader, model factory and `TrainLoop` (subclassed, MDM files untouched) with:

```
--save_dir save/E003_mdm50_baseline --dataset humanml --diffusion_steps 50 --seed 10
--num_steps 750000 --save_interval 5000 --log_interval 100 --use_ema
--train_platform_type TensorboardPlatform --overwrite
```

The subclass adds only:
- **Logging** to TensorBoard and `train_log.jsonl`, every 100 steps: loss, loss per diffusion timestep (50 bins, MDM gives 4), gradient norm, weight norm, distance between raw and EMA weights, steps/s, data-loading time vs. step time, GPU memory, elapsed wall-clock time.
- **Exact stop at `num_steps`, also after a resume.** MDM's loop counts epochs, and after a resume it runs the full epoch count again (this is probably how a 600k config produced a 750k checkpoint).
- **Run record** at start (`run_info.json`): our git commit, MDM commit, package versions, GPU, full arguments, start time.

Resume after a crash: same command; MDM picks up the newest checkpoint, at most 5k steps (about 20 minutes) are lost.

Disk: 151 checkpoints × about 0.31 GB (raw + EMA + optimizer) ≈ 46 GB of 399 GB free. All are kept until the analysis is signed off.

**Equivalence test before launch** (`check_equivalence.sh`). 200 steps each of: stock `train.train_mdm` twice, our launcher, our launcher with `--use_ema`, and stock with another seed as a yardstick. **Result (2026-11-14, with the kernel fix): all four same-seed runs are bit-identical** (max weight difference 0, max loss difference 0); the other seed differs by 0.14 in weights and 0.34 in loss. Resume and the exact stop at `num_steps` were tested separately on a 400-step run.

### Evaluation process

`experiments/baseline/watch.py` runs in tmux session `e003-eval`, picks up new checkpoints and works through a priority queue (180 jobs; `--list` prints it). Benchmark jobs call the Tier 0 evaluator (`experiments/tier0/eval_tier0.py`, now with `--weights raw|ema`), so numbers are computed by the same code as E002: MDM's evaluator calls, TMR scorer, physics stats.

Per evaluated checkpoint, for raw and EMA weights separately:
- **Benchmark metrics on val:** FID, R-precision top 1/2/3, MM-Dist, Diversity. 1000 motions × 3 replications, MDM's default sampler (CFG 2.5), replication k uses seed 10+k as in E002.
- **Extra metrics on the same samples:** TMR similarity, foot sliding, floating, jerk.
- **Held-out denoising loss** (`val_loss.py`): MDM's training loss in eval mode on all 1530 val items and an equally large fixed train subset, at 10 of the 50 timesteps (2, 7, ..., 47), with fixed captions, crops and noise (`save/loss_sets.pt`). With and, on val, without the caption. Much less noisy than FID, so it may be a better early signal for proxy runs, and the val-train gap shows overfitting. About 1.5 min per checkpoint (a forward pass costs ~1 ms per motion); evaluated every 5k steps to 100k, then on the benchmark schedule.

Schedule (47 checkpoints): every 5k steps to 50k, every 10k to 200k, every 25k to 750k. About 100 s per replication gives roughly 8 h of evaluation in total.

Lower-priority queue, run when the main queue is empty:
- APG (scale 4, η = 0, the E002 sampler) on every 50k checkpoint: does its gain hold along training?
- The standard visual set (8 fixed prompts × 3 seeds) on every 50k checkpoint. **Not implemented yet**; all checkpoints are kept, so it can be added at any time.
- 10 replications instead of 3 at the checkpoints around the candidate proxy lengths, to tighten their intervals.

**Final numbers:** steps 600k and 750k on the **test** split, 20 replications, CFG 2.5, raw weights (and EMA, reported separately). This is the comparison with the released checkpoint.

### Analysis

`experiments/baseline/analyze.py` writes `experiments/baseline/RESULTS.md` and plots, re-runnable at any time during the run:
- Each metric against steps and against training hours, raw vs. EMA, with intervals, and the released checkpoint's value as a reference line.
- For candidate lengths (25k, 50k, 100k, 150k, 200k, 300k, 400k): each metric's relative gap to its final value, and whether FID, R-top3 and MM-Dist are all within **5%** and within **10%** of final (Hieu's criterion, two levels). For each level, the earliest checkpoint from which all later checkpoints stay within it. With 3 replications FID's interval is about ±11% of its final value, so the 5% verdict on FID needs the 10-replication follow-up to be trusted.
- A recommended proxy length, stated with what it rests on and what it does not show (one configuration, one seed).

Training hours are computed from steps and the uncontended speed (measured while no evaluation runs), so the evaluator's GPU share does not distort the time axis.

### Tracking

TensorBoard, served from the run directory by the tmux session `e003-tb` (`http://localhost:6006`; from another machine `ssh -L 6006:localhost:6006 <lab-pc>`). Training scalars come from the launcher (`Loss/*`, `LossByQuartile/*`, `Run/*`). The watcher adds one TensorBoard run per weights/sampler under `eval/tb/` (`Eval/*`, `Physics/*`, `HeldOutLoss/*`), so raw and EMA curves overlay in one chart. Nothing is sent to an external service.

### Documentation

- This file: the decisions and their reasons.
- Vault note `E003 - MDM-50 baseline training run` and a row in `Experiment log.md`, written at launch with the commit and command, filled with results at the end.
- `SETUP.md` section 4.2: reproduce command corrected to the released recipe.

## Expected cost

Measured: about 5.3-5.5 steps/s with the GPU to itself, so about 38-39 h of training for 750k steps, plus roughly 15 h of evaluation work sharing the GPU (benchmark 8.6 h, held-out loss 1.3 h, APG 2.8 h, final test 2.2 h): **roughly 2.2 days**. Training drops to about half speed while an evaluation runs. The GPU should not be used for other jobs during that time.

## What happened at launch (2026-11-14)

1. **First launch (17:18) diverged.** Around step 600 the gradient norm jumped from ~0.4 to over 2000 and the loss went from 0.35 back to ~1.15, where it stayed. Stopped after 3000 steps; its log is kept in `save/E003_attempt1_diverged/`.
2. **It was systematic.** Six 2500-step probes (seeds 10, 11, 12, with and without `--mask_frames`) all diverged between steps 550 and 650 (`stability_probe.sh`).
3. **One sample, wrong arithmetic.** A capture of the first spike (`diag_spike.py`) showed a single sample with gradient norm 812 against a batch median of 0.025, at a normal loss. Replaying that step with identical weights, batch, timesteps and noise gave a gradient norm of ~4000-7700 on the GPU and 0.32 on the CPU (float32 and float64).
4. **Cause.** PyTorch's memory-efficient kernel for `scaled_dot_product_attention` (the default for float32 on CUDA, used by `nn.MultiheadAttention` and `nn.TransformerEncoder`) returns wrong values and gradients on the RTX 5070 Ti with PyTorch 2.7.1 + CUDA 12.8. The first module that deviates from the CPU is layer 0's self-attention. With the kernel off, GPU and CPU agree to 1e-5. `scripts/check_attention_kernel.py` reproduces it with random inputs: the kernel is wrong in float32 for head sizes above 64 (outputs and gradients for 72-128, gradients only for 136-256) and correct for ≤ 64 and in float16/bfloat16. MDM's head size is 512 / 4 = 128. The error grows with the size of the attention logits, which is why training is fine for ~500 steps and then collapses. Write-up for future models: `vault/80 Setup/Attention kernel bug on the lab PC.md`.
5. **Fix.** `torch.backends.cuda.enable_mem_efficient_sdp(False)` in `train_instrumented.py`, `val_loss.py` and `experiments/tier0/eval_tier0.py`; `scripts/fixed_attention.py -m <module>` does the same for stock MDM commands. With it, two seeds trained cleanly for 3000 steps (loss 0.52 → 0.18, no spikes), and runs became bit-reproducible.
6. **Consequence for earlier results.** Sampling on the lab PC used the same kernel. On the released checkpoint the per-step error is 0.005 on average (outputs have scale 1) with ~0.1% of values off by more than 0.1. E002 (Tier 0) was measured this way. Re-evaluated with the fix (`save/tier0_fixed/`, same seeds, test split, 20 replications): MDM default FID 0.552 ± 0.047 (was 0.567 ± 0.049), APG scale 4 FID 0.444 ± 0.047 (was 0.463 ± 0.050); R-precision, MM-Dist, Diversity, TMR similarity and foot sliding unchanged to the second decimal. All shifts are inside the confidence intervals and the E002 conclusions hold.

## Decisions

| Decision | Chosen | Decided by |
|---|---|---|
| Baseline recipe | Released config; no `--mask_frames` | Hieu ("make sure deviations are really needed") |
| Attention kernel | Memory-efficient kernel off in every script | Claude's call: without it nothing trains; to confirm |
| Length | 750k steps, matching the released checkpoint | Proposed by Claude, approved by Hieu |
| EMA | Tracked alongside; raw weights are the headline | Proposed by Claude, approved by Hieu |
| Evaluation | Separate process, val split, dense early schedule | Claude's call |
| Proxy-length criterion | FID, R-top3, MM-Dist all within 5% and within 10% of final (two levels); "reached" = stays within from that checkpoint on | Hieu (levels); Claude ("stays within" rule, three metrics) |
| Sampler for the curve | MDM default CFG 2.5; APG at 50k multiples only | Claude's call |
| Logging backend | TensorBoard + JSONL files, nothing sent to external services | Hieu asked for W&B or TensorBoard; TensorBoard chosen by Claude (no account, stays local) |
| Checkpoints | Every 5k steps, all kept | Claude's call |

## As built

| File | Role |
|---|---|
| `experiments/baseline/run_e003.sh` | starts or resumes the three tmux sessions (`e003-train`, `e003-eval`, `e003-tb`); holds the exact training command |
| `experiments/baseline/train_instrumented.py` | launcher: MDM's `train.train_mdm.main()` with an instrumented `TrainLoop` subclass |
| `experiments/baseline/watch.py` | evaluation queue over checkpoints; writes JSON and TensorBoard |
| `experiments/baseline/val_loss.py` | held-out denoising loss |
| `experiments/baseline/analyze.py` | `RESULTS.md` and plots, including the proxy-length tables |
| `experiments/baseline/check_equivalence.sh` | launcher vs. stock MDM |
| `experiments/baseline/stability_probe.sh`, `stability_report.py`, `diag_spike.py` | the divergence investigation |
| `scripts/check_attention_kernel.py`, `scripts/fixed_attention.py` | GPU attention check and wrapper for stock MDM commands |

Run directory: `baselines/mdm/save/E003_mdm50_baseline/` (= `mdm_assets/save/...`): checkpoints, `train_log.jsonl`, `run_info.json`, `code_snapshot/`, `eval/`.
