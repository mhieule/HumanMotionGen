---
tags: [models, overview]
updated: 2026-11-03
---
# Model landscape: what we can run on 10–12 GB

> [!info] Source
> From my survey of 24 and 31 Oct (project doc "motion-models-12gb"). "Fits 12 GB?" is an **estimate** unless the status says *verified*. No 2025–2026 paper I found trains on a 12 GB card; the closest train on one V100 or RTX 4090 at batch 64, which usually fits 12 GB at batch 32 with gradient accumulation.

**Status values:** `to-try` · `running` · `works` · `failed` · `excluded`

## Model research ([[Meeting 01]])
Can we run it? · What does the paper focus on (generation quality vs. real-time vs. something else)? · Hardware needed (exclude compute-hungry ones) · 2025–2026 first

## Baselines named

| Model                       | Year             | Paradigm                                                  | Fits 12 GB?                                                | Focus                | Status                              | Card          |
| --------------------------- | ---------------- | --------------------------------------------------------- | ---------------------------------------------------------- | -------------------- | ----------------------------------- | ------------- |
| **MDM**                     | 2022 / ICLR 2023 | Diffusion, x₀, Transformer encoder, 263-D                 | Train: yes                                                 | Classic T2M baseline | works (sampling + eval, 2026-11-05) | [[MDM]]       |
| **HY-Motion 1.0** (Tencent) | Dec 2025         | Flow-matching DiT, 1B / 0.46B Lite, Qwen3-8B text encoder | **Inference only, with hacks** (official minimum 24–26 GB) | Large-scale quality  | to-try                              | [[HY-Motion]] |

## Tier 1: trainable on 12 GB (recent, with code)
| Model                 | Date                   | Approach                                                           | Size       | Reported hardware           | Text enc.             | HumanML3D FID            | Status                        |
| --------------------- | ---------------------- | ------------------------------------------------------------------ | ---------- | --------------------------- | --------------------- | ------------------------ | ----------------------------- |
| **CMDM**              | Feb 2026, CVPR 2026    | Causal diffusion forcing (FM) in a causal VAE latent; streaming AR | 38M DiT    | 1× A100, batch 128          | DistilBERT            | 0.068 (263) / 0.078 (67) | to-try                        |
| **MSFlow**            | Sep 2026               | FM in motion space, no AE; zero-shot joint control                 | 68M        | ~7.5 h on 1× A100, batch 64 | DistilBERT            | 0.046 (67-D eval)        | to-try                        |
| **MotionHiFlow**      | Apr 2026, CVPR 2026    | Coarse-to-fine FM in a skeleton-aware VAE latent                   | small DiT  | batch 64                    | CLIP                  | 0.032 (263)              | works (inference, 2026-11-05) |
| **ACMDM**             | May 2025               | FM on absolute joint coordinates; ControlNet                       | S–XL       | S: ~8 h on 1× RTX 4090      | CLIP                  | 0.109 (S, 67)            | to-try                        |
| **SALAD**             | Mar 2025, CVPR 2025    | Skeleton-aware latent diffusion (v-pred)                           | ~10M       | 1× V100                     | CLIP                  | 0.076 (263)              | works (inference, 2026-11-05) |
| **MoMask++**          | Jul 2025, NeurIPS 2025 | Multi-scale RVQ tokens + masked Transformer                        | 13–18M     | 1× V100                     | T5-base               | 0.072 (263)              | to-try                        |
| ScaleMoGen            | May 2026               | Next-scale AR over skeletal-temporal tokens                        | 125M       | 1× RTX 4090                 | T5                    | 0.030 (263)              | code not released             |
| MARDM                 | Nov 2024, CVPR 2025    | Masked AR + diffusion/flow head                                    | ~310M (XL) | batch 64                    | CLIP                  | 0.114 (67)               | to-try                        |
| FloodDiffusion (Tiny) | Dec 2025, CVPR 2026    | Diffusion forcing, streaming, time-varying text                    | tiny       | "can train on 5090" (32 GB) | UMT5-XXL → precompute | 0.057 (263, main)        | to-try                        |

## Tier 2: inference only on 12 GB
| Model | Date | VRAM | Notes | Status |
|---|---|---|---|---|
| Kimodo (NVIDIA) | Mar 2026 | <3 GB with `TEXT_ENCODER_DEVICE=cpu` | Text + keyframe/path constraints; **Unitree G1** checkpoints | to-try |
| ARDY (NVIDIA) | Jul 2026 | LLM text encoder ~14 GB → CPU | Real-time AR diffusion; G1 checkpoints | to-try |
| HY-Motion Lite | Dec 2025 | Peak ~16.5 GB at batch 1 (mostly the text encoder) | → [[HY-Motion]] | to-try |
| TACD students | Oct 2026 | 2.6 / 4.0 GB | Distilled HY-Motion / Kimodo; check whether weights are out | to-try |

## Excluded (compute)
- Training HY-Motion, Kimodo, ARDY
- LLM-based motion models (MotionGPT-style, MotionMillion / Go to Zero, LoM)

## Suggested starting set
- **Diffusion:** MDM (baseline) or SALAD
- **Flow matching:** MotionHiFlow or ACMDM-S
- **AR (later):** CMDM; MoMask/MoMask++ for tokens
- **Robot demo / quality reference:** Kimodo-G1 with the text encoder on the CPU
- One evaluator for everything (decide: 263-D vs. 67-D) → [[Motion evaluation metrics]]

> [!note] FID numbers in this table come from two evaluators (263-D and 67-D) and are **not comparable across them**.
