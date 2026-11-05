---
tags: [concept, research]
---
# Image diffusion techniques to port

> [!abstract] Why this note
> Simon: MDM works well but is old. Take what the last years of **image** diffusion and flow research found, apply it to motion on a **fixed dataset**, and measure what helps. This is the menu; [[Idea - Modernize MDM]] is the plan. Techniques from **video** models (temporal attention, long and streaming generation, keyframe control) are in [[Video diffusion techniques to port]].

## In my words


## Menu
| Technique | Source | What it changes | Motion version | Cost |
|---|---|---|---|---|
| **DiT blocks + AdaLN-Zero** | DiT (Peebles & Xie, 2023) | Timestep/condition through adaptive LayerNorm, zero-initialised gates | Replace MDM's prefix-token encoder | low |
| **Rectified flow / v-target** | Rectified Flow (2023), SiT (2024) | Straight paths, velocity target | MDM → flow matching → [[Flow matching]] | low |
| **Logit-normal timestep sampling** | SD3 (Esser et al., 2024) | More training on mid noise levels | Swap the t-sampler | trivial |
| **Loss weighting / Min-SNR-γ** | Hang et al., 2023 | Balances noise levels | Weight the loss per t | trivial |
| **x-prediction for high-dim data** | JiT (Li & He, 2025) | Predict clean data, not noise | ε vs. x₀ vs. v ablation → [[Prediction targets (epsilon, x0, v)]] | trivial |
| **EDM preconditioning and noise distribution** | EDM (Karras et al., 2022) | Input/output scaling, log-normal σ, Heun sampler | Alternative to the cosine schedule | low |
| **Zero-terminal SNR** | Lin et al., 2024 | Fixes schedules that never reach pure noise | Check MDM's schedule | trivial |
| **QK-norm, RoPE** | SD3, modern LLM/DiT practice | Stable attention; relative positions in time | Temporal RoPE over frames | low |
| **EMA (incl. post-hoc EMA)** | EDM2 (Karras et al., 2024) | Better weights for sampling | Evaluate EMA checkpoints | trivial |
| **Guidance interval** | Kynkäänniemi et al., 2024 | CFG only at mid noise levels | → [[Classifier-free guidance]] | trivial |
| **REPA** (representation alignment) | Yu et al., 2025 | Align hidden states with a pretrained encoder → faster training | Align with a motion encoder (e.g. the evaluator or a motion VAE) | medium |
| **Better samplers** | DPM-Solver, Heun | Fewer steps at the same quality | Measure FID vs. steps | low |
| **Latent diffusion** | LDM / MLD | Diffuse in a compressed space | → [[Latent motion diffusion]] | medium |

## How to test
Same data, same evaluator, same budget (steps × batch), change **one thing per run**, at least 2 seeds for the important rows → [[Experiment log]]
