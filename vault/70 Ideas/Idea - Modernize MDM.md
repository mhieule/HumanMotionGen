---
tags: [idea]
status: proposed
source: "Simon, Meeting 01"
effort: M
---
# Idea: Modernise MDM with recent image-diffusion techniques

## Hypothesis
MDM's architecture and training recipe date from 2022. Swapping in what image diffusion has learned since (DiT + AdaLN-Zero, flow matching / v-target, better timestep sampling and loss weighting, QK-norm and RoPE, EMA, better samplers) will improve FID and R-precision on HumanML3D at the **same compute budget**.

## Why it might work / why not
- + These tricks are mostly domain-agnostic and cheap to implement
- + Recent motion models (MotionHiFlow, ACMDM, HY-Motion) already use DiT + flow matching and beat MDM
- − Motion sequences are small; some tricks only pay off at scale

## Minimal experiment
Fixed data and evaluator (see [[Experiment log]]). Starting from [[E001 - Reproduce MDM on HumanML3D]], add **one change per run**:
1. DiT blocks with AdaLN-Zero conditioning instead of prefix tokens
2. Flow-matching objective (v-target) instead of DDPM x₀
3. Logit-normal timestep sampling
4. QK-norm + temporal RoPE
5. Model size sweep: S / M / L at a fixed number of steps

## Success criterion
A clear FID / R-precision gain over E001 with ≥ 2 seeds; a cumulative "MDM-2026" configuration.

## Links
[[Image diffusion techniques to port]] · [[Video diffusion techniques to port]] · [[MDM]] · [[Flow matching]] · [[Conditioning mechanisms]]
