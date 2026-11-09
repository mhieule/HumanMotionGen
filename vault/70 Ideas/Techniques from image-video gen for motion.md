---
tags: [ideas, roadmap, diffusion, mdm, humanml3d]
created: 2026-11-09
---

# Techniques from image/video generation worth trying for motion (MDM / HumanML3D)

Curated from recent image/video generation work (knowledge up to ~mid-2026, not exhaustively verified against the newest arXiv). Ordered within each group by expected value / cost.

## Suggested order
1. Flow matching + logit-normal timesteps + RoPE
2. Interval CFG / APG (sampling only, no retraining)
3. Dispersive loss vs. REPA (ablation)
4. Contrastive Flow Matching
5. GRPO post-training (physics + TMR rewards)
6. Adversarial fine-tuning
7. Latent diffusion / few-step / streaming — only if speed or long-horizon matters

---

## Training objective
- **Flow matching / rectified flow** — straight noise→data paths; more stable training, 10–50 sampling steps instead of 1000.
- **Logit-normal timestep sampling** — focus training on mid-noise levels. Cheap FID gain.
- **Min-SNR-γ weighting** (if staying on DDPM) — per-timestep loss reweighting; ~3× faster convergence on images. One-line change.
- **Keep x₀ prediction** — MDM already does this; keeps geometric losses easy.
- **Progressive-length training** — short clips first, then longer (motion analog of low→high-res pretraining). Pairs with RoPE.

## Architecture
- **RoPE on frame tokens** — relative temporal positions → better length generalization. Exclude the condition token from rotation (or give it a fixed position).
- **QK-norm + adaLN-Zero** — stability when scaling up.
- **Joint text–motion attention + LLM text encoder (T5/Qwen)** — richer text signal than one CLIP token.

## Losses / representation learning
- **REPA** — align denoiser mid-layer features to a frozen motion encoder (e.g. TMR). Faster convergence on small data.
- **Dispersive loss** (Wang & He, 2025) — pushes apart in-batch representations; no external encoder needed. Cheaper alternative to REPA; ablate against it.
- **Contrastive Flow Matching (ΔFM)** — separates flows for different conditions → better text alignment (R-precision).
- **Adversarial loss on x₀ predictions** (ADD/LADD/APT) — small temporal-conv discriminator on joint velocities; fights over-smoothing. Use as a fine-tuning stage.
- **Pseudo-Huber loss** — robust to mocap/video-extracted outliers; matters more with noisy pretraining data.
- **Perceptual feature loss** — on a motion encoder's features of predicted x₀. Must differ from the evaluation encoder.
- **Auxiliary velocity/acceleration outputs** (VideoJAM-style) — predict them as extra channels, not only as penalties. Small expected gain.
- Skip: temporal warping/flicker losses; raw high-frequency penalties (kill fast motion).

## Sampling / guidance (free, no retraining)
- **Interval CFG** — apply guidance only at mid-noise steps. Keeps diversity (MModality) and smoothness.
- **APG** — scale down the parallel part of the CFG update; high guidance without jitter/exaggeration.
- **Best-of-N with verifier** — score with physics metrics + TMR, keep the best.

## Post-training
- **GRPO with computed rewards** — physics (foot skating, penetration, floating, jerk) + TMR text alignment. Needs no new data.
- **Never reward with the T2M evaluator** — reserve it for evaluation to avoid benchmark hacking. Sanity-check with qualitative samples / a second retrieval model (TMR and T2M share training data).
- **Optional: MotionCritic** (ICLR 2025) — human-preference reward model; check input-format compatibility first.
- Heavier option: physics-simulator tracking reward (PHC-style, Isaac Gym/MuJoCo + SMPL conversion).

## Data
- **LLM/VLM recaptioning** — denser body-part- and timing-detailed text.
- **Pretrain noisy → fine-tune clean** — large video-extracted motion first, then mocap; filter by physics/jitter scores.

## Strategic (depends on goal)
- **Latent diffusion with a causal temporal VAE** — worth it for speed, real-time, long/streaming generation. Costs: VAE reconstruction floor on FID, harder joint-level control/editing, fragile two-stage training on small data. If adopted: mild compression (~4× in time), semantic latent regularizer (VA-VAE / REPA-E), equivariance regularization (mirror, heading rotation — EQ-VAE style), raw-space losses on decoded output.
- **Few-step generation** — MeanFlow / consistency / distillation for real-time use.
- **Diffusion Forcing / Self-Forcing** — unbounded streaming generation without drift.
- **Trajectory-first / hierarchical** — root trajectory or keyframes first, then full body (GMD-style). Good for control.
- **Keyframe → infill** — alternative to Diffusion Forcing for minutes-long motion.

Note: explicit coarse-to-fine *generation* (cascades, VAR, Pyramidal Flow) is not the mainstream SOTA paradigm in image/video; progressive training and implicit diffusion coarse-to-fine are. Motion is short/low-dim, so compute savings matter less.

## Evaluation protocol
- Standard HumanML3D metrics (FID, R-precision, MM-Dist, Diversity, MModality), 20 repeats, mean ± 95% CI, same seeds per ablation.
- The T2M evaluator only covers ≤196 frames → long-sequence claims (RoPE, streaming) need: windowed FID/R-precision on 196-frame windows, physical metrics (foot skate, jerk, penetration, floating), and first-vs-last-window drift.
