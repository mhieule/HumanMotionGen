# Baseline training run (E003) and its harness

Trains MDM-50 with the released recipe and records how every metric develops with training steps, to (1) check that our pipeline reproduces the released checkpoint and (2) decide how short proxy runs can be. Results: [RESULTS.md](RESULTS.md) (generated). Design and decisions: [`docs/superpowers/specs/2026-11-14-mdm-baseline-run-design.md`](../../docs/superpowers/specs/2026-11-14-mdm-baseline-run-design.md).

The same scripts are meant for every later training run: only the training flags change.

## Run, watch, resume

```bash
experiments/baseline/run_e003.sh        # start or resume: training, evaluator, TensorBoard (3 tmux sessions)
tmux attach -t e003-train               # or e003-eval, e003-tb; detach with Ctrl-b d
```

- **TensorBoard:** `http://localhost:6006` on the lab PC, or `ssh -L 6006:localhost:6006 <lab-pc>` first. Training: `Loss/*`, `LossByQuartile/*`, `Run/*`. Evaluation: `Eval/*`, `Physics/*`, `HeldOutLoss/*`, one colour per weights/sampler (`eval/tb/raw_cfg2.5`, `ema_cfg2.5`, ...).
- **Resume after a crash or reboot:** run `run_e003.sh` again. Training continues from the newest checkpoint (at most 5k steps lost) and finished evaluations are skipped. Inside the tmux session training restarts itself up to 5 times.
- **Results so far, at any time:**
  ```bash
  cd baselines/mdm && .venv/bin/python ../../experiments/baseline/analyze.py --run save/E003_mdm50_baseline
  ```
- **Evaluation queue:** `.venv/bin/python ../../experiments/baseline/watch.py --run save/E003_mdm50_baseline --num_steps 750000 --list`

## What gets recorded

| Where (under `baselines/mdm/save/E003_mdm50_baseline/`) | What | How often |
|---|---|---|
| `train_log.jsonl`, TensorBoard | loss, loss per diffusion timestep (50 bins) and quartile, gradient norm, weight norm, raw-vs-EMA distance, steps/s, data time, GPU memory, whether an evaluation shared the GPU | every 100 steps |
| `model*.pt`, `opt*.pt` | raw + EMA weights, optimizer state (0.31 GB per checkpoint, all kept) | every 5k steps |
| `eval/val_loss/*.json` | held-out denoising loss on val and on a train subset, 10 timesteps, with and without caption | every 5k to 100k, then as below |
| `eval/val/*_cfg2.5.json` | FID, R-precision, MM-Dist, Diversity, TMR similarity, physics; 1000 motions × 3 replications; raw and EMA | every 5k to 50k, 10k to 200k, 25k to 750k |
| `eval/val/*_apg4.json` | same with the E002 sampler (APG scale 4) | every 50k |
| `eval/test/*.json` | final numbers, 20 replications | 600k and 750k |
| `run_info.json`, `code_snapshot/` | commits, package versions, arguments, the scripts as launched | per start |

## Files

| File | Role |
|---|---|
| `run_e003.sh` | the exact training command; starts the tmux sessions |
| `train_instrumented.py` | launcher: calls MDM's own `train.train_mdm.main()` with a `TrainLoop` subclass that only adds logging, an exact stop at `--num_steps` (also after a resume) and the run record |
| `watch.py` | evaluation queue over checkpoints, in a separate process from training |
| `val_loss.py` | held-out denoising loss (`--build` once creates the fixed sets in `save/loss_sets.pt`) |
| `analyze.py` | `RESULTS.md` + `plots/`, including the proxy-length tables |
| `check_equivalence.sh` | 200-step runs: stock MDM vs. the launcher. Result: bit-identical weights, also with `--use_ema` |
| `stability_probe.sh`, `stability_report.py`, `diag_spike.py` | tools from the divergence investigation (below) |

## The attention-kernel problem on the lab PC

PyTorch's memory-efficient attention kernel returns wrong values and gradients on the RTX 5070 Ti (PyTorch 2.7.1 + CUDA 12.8) for float32 attention with head size > 64 (MDM: 128). Write-up and checklist for other models: `vault/80 Setup/Attention kernel bug on the lab PC.md`. Stock MDM training diverges after about 600 steps for every seed, and sampling is slightly off. Every script here and `experiments/tier0/eval_tier0.py` switch that kernel off; MDM's own commands do not. On this machine run them as

```bash
.venv/bin/python ../../scripts/fixed_attention.py -m sample.generate --model_path ...
```

Check a machine with `baselines/mdm/.venv/bin/python scripts/check_attention_kernel.py`. The investigation is written up in the design document ("What happened at launch").
