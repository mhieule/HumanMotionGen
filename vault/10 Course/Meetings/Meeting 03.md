---
tags: [meeting]
date: 
course: Practical Course
with: Simon Giebenhain
---
# Meeting 03: setup, experiment plan, first sampling results

## Agenda
1. Setup and baseline
2. Directions and experiment setup
3. First results: sampling-only improvements on MDM

---

## 1. Setup and baseline

**Lab PC:** access set up; RTX 5070 Ti (16 GB). MDM's original environment (PyTorch 1.7 / CUDA 11) has no kernels for this GPU, so we wrote a Docker-free setup script: Python 3.11, PyTorch 2.7.1 + CUDA 12.8 (`scripts/setup_mdm.sh`, documented in `SETUP.md`). A fresh machine goes from clone to training in one script.

**Data and evaluation pipeline:**
- HumanML3D, processed 263-D features: 29,228 motions with captions; standard train / val / test splits.
- Evaluator: Guo et al. 263-D text–motion evaluator (FID, R-precision, MM-Dist, Diversity). MDM's download script ships an epoch-11 matching model whose numbers are not comparable with papers; we use the standard epoch-28 one.
- Check: ground-truth row reads R-top1 0.511 / MM-Dist 2.97 as expected, and our evaluation of the released MDM-50 is within the CI of the authors' log (FID 0.53 vs. 0.49).

**Training cost** (1000-step run on the lab PC, MDM-50 default config, batch 64, extrapolated):

| | Lab PC (RTX 5070 Ti) | Laptop (RTX 3060), for reference |
|---|---|---|
| Full MDM run (600k steps) | **~1.5 days** | ~3.2 days |
| Short proxy run (100k steps) | ~6 h | ~13 h |

**Takeaway:** from mid-November to the end of January (~74 days) the lab PC fits at most **~49 full runs** back to back, shared between two people and before evaluation and failed runs. Full runs are therefore kept for the baseline and the final model; comparisons use short proxy runs (~4 per day).

---

## 2. Directions and experiment setup

**Goal:** improve MDM with techniques from recent image/video generation, measured on a fixed dataset and evaluator. Menus: [[Image diffusion techniques to port]], [[Video diffusion techniques to port]], [[Techniques from image-video gen for motion]].

**Protocol** (following the points from [[Meeting 02]]):
- **Held-out selection:** all choices are made on the **val** split; the **test** split is used only for final numbers.
- **Short proxy runs** (~100k steps) instead of full runs; one change at a time, 2 seeds; a change is kept only if the gain exceeds the seed-to-seed spread.
- **Proxy check first:** baseline at proxy length with 3 seeds (noise level), and checkpoints of one full run to see whether early rankings predict final rankings.
- **Report performance against training time** (and sampling cost), not only final metrics.
- **Extra metrics** beside FID / R-precision: physics (foot sliding, floating, jerk) and a second text–motion model (TMR). Neither the evaluator nor its training data is used for training or selection signals.

**Order, cheapest first:**

| Tier | What | Cost | Status |
|---|---|---|---|
| 0 | Sampling only: guidance scale, interval CFG, APG, best-of-N | no training | **done** (section 3) |
| 1 | Training recipe: flow matching (x₀ vs. v vs. ε target), logit-normal timesteps, RoPE + QK-norm, adaLN-Zero / DiT blocks | proxy runs | next |
| 2 | Auxiliary losses: dispersive loss vs. REPA, contrastive flow matching | proxy runs | |
| 3 | Post-training: GRPO with physics + text-alignment rewards | fine-tuning | |

Final: one full-length run of the best configuration over the Christmas break, plus remove-one ablations.

**Points to discuss:**
- Work split between us (e.g. diffusion line vs. flow matching line on the shared pipeline).
- Main metric: we currently select by FID, with R-precision not allowed to drop. Is that the right choice?
- Text encoder: CLIP gives one vector per caption ([[Meeting 02]]). Test a token-level encoder as a Tier 1 item?

---

## 3. First results: sampling-only improvements on MDM

**Setup:** released MDM-50, no retraining; only the sampler changes. 24 sampler configurations compared on **val** (5 replications); the best of each method then evaluated on **test** with the standard protocol (1000 motions × 20 replications, mean ± 95% CI). Best-of-N picks candidates with TMR, never with the evaluator. Details: [[E002 - Inference-time guidance on MDM (Tier 0)]].

**Test split, 20 replications:**

| Sampler | FID ↓ | R-top3 ↑ | MM-Dist ↓ | Diversity → | Sampling cost |
|---|---|---|---|---|---|
| Real motions | ~0 | 0.799 | 2.973 | 9.53 | |
| MDM default (CFG 2.5) | 0.567 ± 0.049 | 0.750 | 3.338 | 9.84 | 1× |
| **APG** (scale 4, η = 0) | **0.463 ± 0.050** | 0.753 | 3.310 | 9.70 | 1× |
| Interval CFG (best variant) | 0.593 ± 0.051 | 0.752 | 3.343 | 9.84 | 0.9× |
| APG + best-of-4 (TMR) | 0.374 ± 0.031 | 0.805 | 2.977 | 9.77 | 4.6× |

**Observations:**
- **APG** lowers FID by ~18% at no extra cost; text alignment unchanged.
- **Guidance scale** is already well tuned: scales 1.5–4 give the same FID within the CI.
- **Interval CFG** (guidance only at mid noise, which helps image models) does not transfer: removing guidance from the noisiest steps lowers R-top3 from 0.75 to ~0.69. MDM seems to fix the action in the first few steps.
- **Best-of-4** improves all metrics, but R-top3 and MM-Dist reach the level of real motions. TMR and the evaluator were trained on the same data, so part of this gain may be agreement between the two models. To check: select with a TMR trained on KIT-ML instead, and inspect samples visually.
- **No sampler reduces foot sliding** (~7 cm/s for all configurations vs. 4.8 for real motions). This needs changes at training time (Tier 1–3).

**Next:** baseline training run on the lab PC, proxy check, then Tier 1.

---

## Notes

## Decisions

## Action items
- [ ] 
