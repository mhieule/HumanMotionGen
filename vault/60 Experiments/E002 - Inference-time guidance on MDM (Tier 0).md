---
tags: [experiment]
id: E002
date: 2026-11-10
status: done
model: MDM
dataset: HumanML3D
representation: 263-D
target: x0
baseline: "E001 pretrained MDM-50, CFG 2.5"
commit: "uncommitted (experiments/tier0/)"
config: "experiments/tier0/pipeline.py"
gpu_hours: 9.7
fid: 0.463
r_precision_top3: 0.753
---
# E002: Inference-time guidance on MDM (Tier 0)

## Question / hypothesis
Can sampling-only tricks from image diffusion (guidance-scale tuning, interval CFG, APG, best-of-N with a verifier) improve the **pretrained** MDM-50 on HumanML3D, without retraining? Tier 0 of [[Techniques from image-video gen for motion]].

## Setup
- Model: released MDM-50 (`humanml_enc_512_50steps/model000750000.pt`), no retraining. Baseline = MDM's default sampler (CFG 2.5, 100 network passes per motion).
- Code: `experiments/tier0/` (README there). Guidance wrapper replaces MDM's `ClassifierFreeSampleModel`; plain CFG reproduces it (checked within MDM's GPU run-to-run noise).
- Best-of-N verifier: **TMR** text–motion similarity (optionally + physics rank: foot slide, floating, jerk). Never the T2M evaluator.
- Protocol: **stage 1** = 24 configs on the **val** split, 5 reps → per method, lowest FID with R@3 ≥ baseline − CI. **Stage 2** = winners on **test**, 20 reps (standard). Replication k uses seed 10+k for every config.
- Hardware: lab PC, RTX 5070 Ti. 1000 motions ≈ 97 s for 1-candidate configs. Whole pipeline 9.7 h.

## Results (test split, 20 reps)
| Run | FID ↓ | R-Prec top-3 ↑ | MM-Dist ↓ | Diversity → | Cost | Notes |
|---|---|---|---|---|---|---|
| Real motions | ~0 | 0.799 | 2.973 | 9.53 | | reference |
| baseline: MDM, CFG 2.5 | 0.567 ± 0.049 | 0.750 | 3.338 | 9.84 | 1× | E001 measured 0.527 ± 0.047 (other seeds) |
| **APG, scale 4, η = 0** | **0.463 ± 0.050** | 0.753 | 3.310 | 9.70 | **1×** | −18% FID, free |
| Interval CFG, steps 8–49, scale 3.5 | 0.593 ± 0.051 | 0.752 | 3.343 | 9.84 | 0.92× NFE | no gain |
| APG + best-of-4 (TMR) | 0.374 ± 0.031 | 0.805 | 2.977 | 9.77 | 4.6× | R@3 / MM-Dist at the real-data level, see caveat |

Full tables incl. the 24-config val sweep, R@1, TMR similarity and physics: `experiments/tier0/RESULTS.md`.

## Samples
<!-- TODO: same prompts, baseline vs. APG s4 vs. APG+best-of-4; check whether best-of-4 looks better or only scores better -->

## Observations
- **Guidance scale is already tuned.** CFG 1.5–4 gives val FID 0.46–0.48, all within the CI; R@3 peaks at 3–3.5 with no FID gain. 5.0 starts to hurt.
- **APG is the free win.** η = 0 (drop the update component parallel to the conditional x₀) at scale 4: val FID 0.382 vs 0.460, test 0.463 vs 0.567, alignment unchanged, diversity slightly lower (9.70 vs 9.84; real 9.53). Negative momentum (−0.5) did not help; scale 8 breaks down (FID 0.85). η = 0.5 sits in between.
- **Interval CFG does not transfer to MDM.** Every interval that drops guidance on the noisiest steps (t ≥ 44, σ > 5.3) loses a lot of alignment (val R@3 ≈ 0.68–0.69 vs 0.75). Dropping only the 8 least-noisy steps is neutral. In images the opposite holds (guidance at high noise hurts diversity); for MDM the action seems to be decided in the first few steps.
- **Best-of-N (TMR) mainly buys text alignment**: val R@3 0.791 vs 0.752, TMR similarity 0.548 vs 0.433, FID 0.397. Adding physics to the verifier lowers foot sliding a bit (6.4 vs 7.0 cm/s) but gives back the FID gain. N = 8 ≈ N = 4 at twice the cost.
- **Caveat on best-of-N:** combined with APG it reaches R@3 0.805 and MM-Dist 2.977, i.e. *above* real motions (0.799 / 2.973). TMR and the T2M evaluator were trained on the same HumanML3D train split, so part of this is the verifier agreeing with the evaluator, not better motion. Needs a visual check and ideally a retrieval model with other training data.
- **Physics is not fixed by any sampler**: foot sliding 7.0–7.4 cm/s for every config vs 4.8 for real motions. This needs training-side changes (Tier 1–3, e.g. GRPO with physics rewards).

## Conclusion / next
- Use **APG (scale 4, η = 0)** as MDM's default sampler from now on: −18% FID at no cost. Re-check on retrained models (Tier 1), since the best scale may move.
- Report best-of-N separately with its 4.6× cost and the evaluator-agreement caveat.
- Next: visual comparison of samples; then Tier 1 (flow matching, logit-normal timesteps, RoPE) on the proxy protocol.
