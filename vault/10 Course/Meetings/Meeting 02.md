---
tags: [meeting]
date: 
course: Practical Course
with: Simon Giebenhain
---
# Meeting 02: survey results



## Agenda / slides
1. **Model landscape:** a table of open-source models with code, ranked by how feasible they are on 12 GB → [[Model landscape]]
2. **Data:** shortlist with size, access, text quality, scene geometry → [[Dataset overview]]
3. **What already runs:** MDM, SALAD, MotionHiFlow → [[MDM]], [[HY-Motion]]
4. **Questions** → [[Questions for Simon]]

## 1. Model Landscape: [[Model landscape]]

### Models to compare with MDM
**Criteria:** pretrained HumanML3D checkpoint released · uses the same Guo et al. 263-D evaluator as MDM, so the numbers are directly comparable · runs on the RTX 3060 (12 GB) ·

| Pick  | Model                        | Why                                                                                                                                                 |
| ----- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **MotionHiFlow** (CVPR 2026) | Flow matching + DiT in a VAE latent + hierarchical: the "modern" counterpart to MDM. FID 0.032 vs. MDM-50 0.489                                     |
| **2** | **SALAD** (CVPR 2025)        | Still *diffusion*: isolates what modern diffusion tricks buy over MDM. ~10M params<br>Modern diff tricks: **(v-pred, skeleton-aware latent, DDIM)** |
| later | MoMask                       | Masked tokens (AR-like), mature repo                                                                                                                |
| later | CMDM (CVPR 2026)             | Causal/streaming AR; uses 67-D / 259-D evaluators, **not** MDM's 263-D; repo marked "temporary release"                                             |

**Proposal:** MDM (diffusion, old) vs. SALAD (diffusion, modern) vs. MotionHiFlow (flow matching, modern), all with the same 263-D evaluator and the same prompts → a 3-row table plus side-by-side videos. That also sets up the diffusion vs. flow matching work split.

## 2. Dataset: [[Dataset overview]]
Focus on **HumanML3D**:
- Baseline for MDM and a lot of models
- Focus on "Text + motion" pairs
- Quite large

## 3. Results so far (2026-11-05)
### Side by side: MDM vs. SALAD vs. MotionHiFlow
All three run locally on the RTX 3060. Same 6 prompts, 6 s each, 3 seeds, each model at its paper's sampler settings (MDM 50 DDPM steps, SALAD 50 DDIM steps, MotionHiFlow 12 flow steps). Rows = models, columns = seeds. Re-run with `scripts/compare/compare.sh <name> <prompts.txt>`.


![[grid_p00.mp4]]
![[grid_p01.mp4]]
![[grid_p02.mp4]]
![[grid_p03.mp4]]
![[grid_p04.mp4]]
![[grid_p05.mp4]]




**Physics check** (`scripts/compare/physics_table.py`, 18 clips per model, raw network output without foot IK; a sanity check, not a benchmark):

| Model | floating % | foot slide cm/s | slide > 10 cm/s % | jerk m/s³ |
|---|---|---|---|---|
| Real (300 HumanML3D test clips) | 7 | 4.7 | 12 | 52 |
| MDM | 5 | 8.5 | 30 | 55 |
| SALAD | 4 | 14.7 | 63 | 57 |
| MotionHiFlow | 3 | 10.6 | 36 | 90 |

### First qualitative impressions:
- All three slide their feet more than real motion. The newer models are **not** better here: SALAD slides most, MotionHiFlow is the most jittery. FID doesn't capture this → **argument for more physics/VLM metrics** in our evaluation.
- MDM's three seeds look almost the same, while SALAD and MotionHiFlow vary more between seeds.

### MDM evaluation, our run (standard 263-D evaluator)
Pretrained `humanml_enc_512_50steps`, `wo_mm` protocol (1000 test prompts × 20 replications, mean ± 95% CI), RTX 3060, ~45 min. **Reproduced**: every metric is within the confidence interval of the authors' log.

| | R-top1 ↑ | R-top3 ↑ | FID ↓ | Matching ↓ | Diversity → |
|---|---|---|---|---|---|
| Real motions (our ground-truth row) | 0.511 | 0.798 | 0.002 | 2.970 | 9.458 |
| MDM-50, authors' log | 0.455 ± 0.007 | 0.749 ± 0.006 | 0.489 ± 0.047 | 3.326 ± 0.025 | 9.924 ± 0.083 |
| **MDM-50, our run** | **0.453 ± 0.005** | **0.749 ± 0.006** | **0.527 ± 0.047** | **3.328 ± 0.025** | **9.949 ± 0.090** |

→ The evaluation pipeline is trustworthy, so it's ready to score SALAD, MotionHiFlow and our own models. 


## 4. Struggles / questions:
### Question 1: 
**Porting advanced image/video gen techniques to classical models \(MDM\)**: Could we do this on different levels? 
  - **High-level:** leveraging the generative priors of these advanced models directly:
    - Could be something similar to https://marcb.pro/atu/: 
      - From a \(textual\) condition -> video gen -> video motion tracking -> motion
  - **Low-level:** Cherry picking the techniques, apply onto parts of MDM
    - Example: diffusion on the latent space only, diffusion forcing, etc.
- Intuitively how could I judge that one cherry-picked technique is better for MDM baseline. Do you have any tips?
### Question 2: 
 **Use of VLM:** I have looked into the related papers, even the latest ones do not employ VLM. My questions are:
  - Intuitively I reckon VLM could validate our output more precisely than the traditional probabilistic evaluators. It could answer questions like:
    - Are the physical constraints held?
    - Does the generated motion follow the condition closely? Are there redundant motions?
  - Does VLM as critic make sense from your perspective?
  - We could also employ it as part of the loss, but I myself am sceptical because \(a\) the signal might not be differentiable \(b\) the generator would possibly learn to produce adversarial examples instead of correct motions


## Notes

**Hardware:** RTX 5070 Ti, sudo rights.

**Experiment strategy (like companies do):**
- Start with small models so experiments are fast and we can train several times a week; scale down (e.g. sparse attention) if needed to fit.
- Estimate rough scaling laws from the small models, then train the best-effort model at the end of the semester. The scaling laws need not be precise; what matters is a fair comparison: held-out val set, fixed val metrics, no data leaking between training and evaluation.
- Plot **performance (y) against training time (x)** and decide from that.
- Judge efficiency too: a loss can help but be expensive to compute. We want a quantifiable measure of a method's efficiency.
- Train from scratch first; fine-tune later only if it makes sense.
- Ignore inference-only methods and focus on models we can train: we learn more of the important details that way.

**Evaluation metrics (open problem):**
- Nail down which metrics we compare against the baseline, and whether we can afford to compute them. Check which metrics papers use.
- FID is expensive: how many generated and ground-truth samples are needed? Check the literature, also for alternatives and new metrics.
- Test prompts must be challenging. Quantify it, e.g. as hard as the hardest 5% of dataset prompts; also check how often a prompt repeats in the dataset.
- VLMs (vision-language models): use only for evaluation.

**Text encoder:** CLIP compresses a text into a single vector, which may be too coarse. Look at the HumanML3D captions to judge this, and measure how swapping the encoder changes performance. SnapMoGen is similar to HumanML3D but has longer texts, so it can test the encoder.

**Architecture:** diffusion trends start in image/video generation and later reach niches like motion generation.

**Models to start from:**
- **SALAD:** good starting point, few tricks.
- **MotionHiFlow:** a hierarchy plus latents (e.g. a ~100-frame sequence split into 20 chunks, one latent per chunk) cuts training compute. Use it only if we can explain why the hierarchy helps.
- Start with text → motion; add scene conditioning (motion + scene + text) later.

**Losses:**
- Closed-source models as a loss: not differentiable, usable only via RL (reinforcement learning).
- Open models as a loss: expensive, try later.

**High-level world models (Q1):** too expensive for the course, and physical constraints do not hold. Possibly useful for data generation.

## Action items
- [ ] Train MDM, SALAD and MotionHiFlow: where we are, how long it takes, which parameters
- [ ] Fix the dataset: HumanML3D plus a set of complicated prompts
- [ ] Research FID: how many samples are needed, how expensive it is, and alternatives
- [ ] Look at new papers for new metrics
- [ ] Plot methods side by side (e.g. MotionHiFlow architecture vs. SALAD loss); they may turn out similar, or show useful failure cases
- [ ] Send results on Teams

## Decisions
- Fixed dataset: HumanML3D with complicated prompts.
- Train from scratch first, fine-tune later.
- Focus on trainable models; skip inference-only ones.
- Small models and fast iteration first, best-effort model at the end of the semester.
