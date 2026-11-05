---
tags: [idea]
status: proposed
source: "Simon, Meeting 01"
effort: S
---
# Idea: Prediction target study (ε vs. x₀ vs. v)

## Hypothesis
For motion, the choice of target matters as much as for images. Simon's intuitions: ε is hard at low noise but learns the high frequencies, and v is always hard. JiT (2025) suggests x-prediction wins for data on a low-dimensional manifold. Matching the loss weights should separate the effect of the **parametrisation** from the effect of the **weighting**.

## Minimal experiment
Same network, data and steps. Grid:
- Target ∈ {ε, x₀, v}
- Loss weighting ∈ {native, matched to the x₀-loss (SNR-based)}
- Read out: FID / R-precision, **loss per noise-level bucket**, the denoising trajectory, foot skating

## What would be interesting
- Does ε give less jitter (high frequencies) but worse global structure?
- Do geometric (FK) losses only help with x₀ / v?

## Links
[[Prediction targets (epsilon, x0, v)]] · [[Training loss vs sample quality]] · [[Idea - Modernize MDM]]
