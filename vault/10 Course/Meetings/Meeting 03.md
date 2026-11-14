---
tags: [meeting]
date: 
course: Practical Course
with: Simon Giebenhain
---
# Meeting 03: setup, experiment plan, first sampling results

**Main abbreviations:** MDM = Motion Diffusion Model (the baseline we improve). CI = confidence interval. Evaluation metrics are explained in section 1.

## Agenda
1. Setup and baseline
2. Directions and experiment setup
3. First results: sampling-only improvements on MDM

---

## 1. Setup and baseline

**Lab PC:** access set up; RTX 5070 Ti (16 GB). MDM's original environment (PyTorch 1.7 / CUDA 11) has no kernels for this GPU, so we wrote a Docker-free setup script: Python 3.11, PyTorch 2.7.1 + CUDA 12.8 (`scripts/setup_mdm.sh`, documented in `SETUP.md`). A fresh machine goes from clone to training in one script.

**Data and evaluation pipeline:**
- HumanML3D, processed 263-D features: 29,228 motions with captions; standard train / val / test splits.
- Evaluator: Guo et al. 263-D text–motion evaluator. It embeds motions and texts and reports:
  - **FID (Fréchet Inception Distance):** distance between the feature distributions of generated and real motions; lower = more realistic.
  - **R-precision (top-1 / top-3):** how often the correct text ranks in the top 1 or top 3 among 32 candidate texts for a motion; higher = better text match.
  - **MM-Dist (Multimodal Distance):** average distance between a motion and its text in feature space; lower = better.
  - **Diversity:** average distance between random pairs of generated motions; should be close to that of real motions.
  MDM's download script ships an epoch-11 matching model whose numbers are not comparable with papers; we use the standard epoch-28 one.
- Check: ground-truth row reads R-top1 0.511 / MM-Dist 2.97 as expected, and our evaluation of the released MDM-50 is within the CI of the authors' log (FID 0.53 vs. 0.49).

**Training cost** (1000-step run on the lab PC, MDM-50 default config, batch 64, extrapolated):

|                              | Lab PC (RTX 5070 Ti) | Laptop (RTX 3060), for reference |
| ---------------------------- | -------------------- | -------------------------------- |
| Full MDM run (600k steps)    | **~1.5 days**        | ~3.2 days                        |
| Short proxy run (100k steps) | ~6 h                 | ~13 h                            |

**Takeaway:** from mid-November to the end of January (~74 days) the lab PC fits at most **~49 full runs** back to back. 
Full runs are therefore kept for the baseline and the final model; comparisons use short proxy runs (~4 per day).

---

## 2. Directions and experiment setup

**Goal:** improve MDM with techniques from recent image/video generation, measured on a fixed dataset and evaluator. Menus: [[Image diffusion techniques to port]], [[Video diffusion techniques to port]], [[Techniques from image-video gen for motion]].

**Protocol** (following the points from [[Meeting 02]]):
- **Held-out selection:** all choices are made on the **val** split; the **test** split is used only for final numbers.
- **Short proxy runs** (~100k steps) instead of full runs; one change at a time, 2 seeds; a change is kept only if the gain exceeds the seed-to-seed spread.
- **Proxy check first:** baseline at proxy length with 3 seeds (noise level), and checkpoints of one full run to see whether early rankings predict final rankings.
- **Report performance against training time** (and sampling cost), not only final metrics.
- **Extra metrics** beside FID / R-precision: physics (foot sliding, floating above the ground, jerk = sudden changes in acceleration) and a second text–motion model, TMR (Text-to-Motion Retrieval), which scores how well a motion matches its text. 

**Order, cheapest first** (priorities revised after the meeting; see Decisions):

| Tier | What                                                        | Cost        | Status               |
| ---- | ----------------------------------------------------------- | ----------- | -------------------- |
| 0    | Sampling only (change how we generate, not the model)       | no training | **closed** (section 3; result depends on the model) |
| 1    | Training recipe (change how the model is trained and built) | proxy runs  | **focus**, next      |
| 2    | Auxiliary losses (extra training objectives)                | proxy runs  | **focus**            |
| 3    | Post-training (fine-tune the trained model with rewards)    | fine-tuning | later, needs a good model |

**Tier 0: sampling only**
- **Guidance scale:** how strongly generation is pushed toward the text prompt.
- **CFG (Classifier-Free Guidance):** at each step, mix the model's prediction with and without the text; the guidance scale sets the mix.
- **Interval CFG:** apply CFG only at mid noise levels instead of at every step.
- **APG (Adaptive Projected Guidance):** a CFG variant that removes the part of the guidance update that oversaturates or distorts samples, so stronger guidance can be used safely.
- **Best-of-N:** generate N candidates per prompt and keep the one a scoring model rates best.

**Tier 1: training recipe**
- **Flow matching:** train the model to follow a straight path from noise to data, instead of standard diffusion's curved path.
- **x₀ vs. v vs. ε target:** what the network predicts: the clean motion (x₀), the velocity along the path (v), or the added noise (ε).
- **Logit-normal timesteps:** sample training noise levels more often from the middle range, where learning matters most.
- **RoPE (Rotary Position Embedding):** encode frame positions by rotating attention features, which handles relative timing and varying lengths better.
- **QK-norm:** normalize attention queries and keys to stabilize training.
- **DiT (Diffusion Transformer) blocks with adaLN-Zero (adaptive Layer Norm, zero-initialized):** text and timestep set the scale and shift of each layer norm; each block starts as an identity, so training starts stable.

**Tier 2: auxiliary losses**
- **Dispersive loss:** push the internal representations of different samples apart so they do not collapse; needs no external model.
- **REPA (REPresentation Alignment):** align the model's internal features with those of a pretrained encoder.
- **Contrastive flow matching:** also push predictions for different conditions apart, so outputs depend more on the text.

**Tier 3: post-training**
- **GRPO (Group Relative Policy Optimization):** reinforcement learning fine-tuning. Generate a group of samples per prompt, score them with rewards, and push the model toward the samples that beat the group average.
- **Rewards:** physics (e.g. less foot sliding) and text alignment (how well the motion matches the prompt).

Final: one full-length run of the best configuration over the Christmas break, plus remove-one ablations.

---

## 3. First results: sampling-only improvements on MDM

**Setup:** released MDM-50, no retraining; only the sampler changes. 24 sampler configurations compared on **val** (5 replications); the best of each method then evaluated on **test** with the standard protocol (1000 motions × 20 replications, mean ± 95% CI). Best-of-N picks candidates with TMR, never with the evaluator. Details: [[E002 - Inference-time guidance on MDM (Tier 0)]].

**Test split, 20 replications:**

| Sampler                                                              | FID ↓             | R-top3 ↑  | MM-Dist ↓ | Diversity → | Sampling cost |
| -------------------------------------------------------------------- | ----------------- | --------- | --------- | ----------- | ------------- |
| Real motions                                                         | ~0                | 0.799     | 2.973     | 9.53        |               |
| **MDM default (CFG 2.5)**                                            | **0.567 ± 0.049** | **0.750** | **3.338** | **9.84**    | **1×**        |
| **APG** (scale 4, η = 0: the oversaturating component fully removed) | **0.463 ± 0.050** | 0.753     | 3.310     | 9.70        | 1×            |
| Interval CFG (best variant)                                          | 0.593 ± 0.051     | 0.752     | 3.343     | 9.84        | 0.9×          |
| APG + best-of-4 (TMR)                                                | 0.374 ± 0.031     | 0.805     | 2.977     | 9.77        | 4.6×          |

**Observations:**
- **APG** lowers FID by ~18% at no extra cost; text alignment unchanged.
- **Guidance scale** is already well tuned: scales 1.5–4 give the same FID within the CI.
- **Interval CFG** (guidance only at mid noise, which helps image models) does not transfer: removing guidance from the noisiest steps lowers R-top3 from 0.75 to ~0.69. MDM seems to fix the action in the first few steps.
- **Best-of-4** improves all metrics, but R-top3 and MM-Dist reach the level of real motions. TMR and the evaluator were trained on the same data, so part of this gain may be agreement between the two models. To check: select with a TMR trained on KIT-ML (KIT Motion-Language dataset, a different dataset) instead, and inspect samples visually.
- **No sampler reduces foot sliding** (~7 cm/s for all configurations vs. 4.8 for real motions). This needs changes at training time (Tier 1–3).

**Next:** baseline training run on the lab PC, proxy check, then Tier 1.

---

## Notes

**Evaluation:**
- The text–motion evaluator may depend on the method or model being evaluated: check how good it is in its paper and how well it agrees with human perception.
- Keep Diversity to make sure the model does not collapse.
- Find how many samples FID needs. Goal: reliable feedback that is cheap enough to evaluate often.

**Proxy runs:**
- Check whether 100k steps makes sense; tune the length if needed. Save checkpoints more often to see how the model behaves over training.
- Ideally run some evaluations during training, not only at the end.
- Early on, be very efficient: watch how the model behaves across steps, including the training curve.
- Worth running one full run once, logged in Weights & Biases.

**Tier 0:** what works best depends on the model, so stop here and focus on Tiers 1 and 2.

**Fundamental aspects to look at (Architecture + Representation + Data):**
- **Representation:** joint angles/velocities, or grouping timesteps into one latent to reduce complexity. A representation can be intuitive for humans but hard for the model to learn or give poor gradients; the right one makes learning easier.
- **Text encoder:** is it good enough or too coarse? A better encoder could mean better and faster training.
- **Data:** generate our own captions.
- Joint angles and noise may interact badly: interval CFG and logit-normal timesteps could produce strange joints or motion.

**Tier 1 order (fundamentals first, details later):**
1. Latent representation: most important and trickiest. Take inspiration from MotionHiFlow.
2. Flow matching vs. diffusion.
3. Learning rate: also important.
4. QK-norm: later, only if training becomes unstable.

**Tier 3:** physics as rewards could be interesting, but it needs a good model first.

**Overall:** the current plan is good; try it out first.

## Decisions
- Tier 0 is closed: keep APG as the default sampler and don't tune samplers further.
- Tiers 1 and 2 are the main focus for the course. Tier 3 only once we have a good model.
- Tier 1 starts with the high-impact changes: latent representation, then flow matching vs. diffusion, then learning rate; QK-norm only if needed.
- Log training to Weights & Biases.

## Action items
- [ ] Set up Weights & Biases logging
- [ ] Do one full MDM run logged in W&B, with frequent checkpoints
- [ ] Check whether 100k-step proxy runs make sense (compare checkpoint rankings with the full run)
- [ ] Add evaluation during training (cheap metrics at each checkpoint)
- [ ] Find how many samples FID needs (reliability vs. cost)
- [ ] Check how well the evaluator agrees with human perception (paper numbers)
- [ ] Write a script to run many configurations
- [ ] Tier 1: latent representation (MotionHiFlow-style), flow matching vs. diffusion, learning rate
- [ ] Check whether the CLIP text encoder is too coarse
- [ ] Look into generating our own captions
