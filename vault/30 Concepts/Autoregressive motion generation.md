---
tags: [concept, paradigm]
---
# Autoregressive motion generation

> [!abstract] Primer (replace with your own words)
> Generate the motion piece by piece, each conditioned on what came before, like an LLM: $p(x) = \prod_i p(x_i \mid x_{<i}, c)$. With **discrete tokens**, a VQ-VAE first turns motion into codebook indices, and a Transformer then predicts them (left to right as in T2M-GPT, or by masked filling as in MoMask). With **continuous** outputs, each chunk is produced by a small diffusion or flow head (MARDM), or by diffusion forcing (CMDM, DART, FloodDiffusion). Main risk: errors compound over long sequences (exposure bias).

## In my words


## Key ideas
- **VQ-VAE:** encoder → nearest codebook entry → decoder; commitment loss. The decoder caps the achievable quality.
- **Residual VQ** (MoMask): several codebooks, each quantising what the previous one left over.
- **Causal vs. masked:** GPT-style left to right, vs. BERT-style parallel filling over a few iterations.
- **Tokenising the body** (Simon): split a pose into body-part tokens instead of one global latent; this matters for AR → [[Motion representations]]
- **Streaming / real-time:** AR suits online control (robots), e.g. CMDM, ARDY, FloodDiffusion.

## Why it matters here
- Simon: **later in the semester**; diffusion and flow matching come first ([[Meeting 01]]).
- A candidate for the teammate's paradigm if we split by paradigm → [[Roadmap]]

## Seen in
- T2M-GPT (2023), MoMask (CVPR 2024), MoMask++ (2025), MARDM (CVPR 2025), DART (2025), CMDM (CVPR 2026), ScaleMoGen (2026)
