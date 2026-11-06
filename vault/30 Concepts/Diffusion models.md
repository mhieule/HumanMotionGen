---
tags: [concept, paradigm]
---
# Diffusion models

> [!abstract] Primer (replace with your own words)
> Diffusion models learn to reverse a fixed noising process. During training, take a real motion, add Gaussian noise at a random level t, and train a network to undo it in **one step**. During sampling, start from pure noise and apply the network **repeatedly** (DDPM uses ~1000 steps; DDIM or ODE solvers use ~20–50), each time moving to a slightly less noisy sample. The condition (e.g. text) is an extra input to the network.

## In my words


## Key equation / picture
- Forward (noising): $x_t = \alpha_t x_0 + \sigma_t \varepsilon,\ \varepsilon \sim \mathcal N(0, I)$, where the schedule takes $\alpha_t$ from 1 down to ≈0 (MDM uses a cosine schedule).
- Training: $\mathcal L = \mathbb E_{x_0, t, \varepsilon}\, w(t)\, \lVert f_\theta(x_t, t, c) - \text{target} \rVert^2$, where the target is ε, x₀ or v → [[Prediction targets (epsilon, x0, v)]]
- Sampling: iterate $x_T \to x_{T-1} \to \dots \to x_0$; deterministic (DDIM / ODE) or stochastic (DDPM / SDE).

## Why it matters here
- **MDM** is a diffusion model: a Transformer encoder over frames that predicts x₀ → [[MDM]]
- Train a single step, sample in ~20 steps → errors accumulate; the training loss is a weak signal → [[Training loss vs sample quality]]
- Diffusion and flow matching are close relatives; one can be rewritten as the other → [[Flow matching]]

## Check yourself
- Why can't a regressor map text → motion directly? (One text fits many motions; regression collapses to the blurry average.)
- What does the network see at each step? (The noisy motion x_t, the timestep t, the condition c.)

## Seen in
- MDM (Tevet et al., ICLR 2023), MLD (Chen et al., CVPR 2023), SALAD (CVPR 2025)
