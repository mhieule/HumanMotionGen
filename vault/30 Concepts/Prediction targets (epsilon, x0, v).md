---
tags: [concept, training]
---
# Prediction targets (epsilon, x0, v)

> [!abstract] Primer (replace with your own words)
> The network can output the noise ε, the clean sample x₀, or the velocity v. Given $x_t$, any one of them determines the other two, so they are different **parametrisations** of the same model. They differ in **how errors are weighted across noise levels** and in how easy the regression is at each level. The loss has the same structure for all three; switching target is like changing the loss weight $w(t)$ (Simon: "both loss: similar structure").

## In my words


## Key equations
Diffusion form $x_t = \alpha_t x_0 + \sigma_t \varepsilon$:
- From ε to x₀: $\hat x_0 = (x_t - \sigma_t \hat\varepsilon)/\alpha_t$ → when $\alpha_t \to 0$ (high noise), small ε errors are **amplified**
- v-prediction (Salimans & Ho 2022): $v = \alpha_t \varepsilon - \sigma_t x_0$
- Flow matching ($\alpha = 1-t,\ \sigma = t$): $v = \varepsilon - x_0$ → [[Flow matching]]
- $\lVert \hat x_0 - x_0 \rVert^2 = (\sigma_t/\alpha_t)^2 \lVert \hat\varepsilon - \varepsilon \rVert^2$, so an ε-loss is an x₀-loss with weight $\text{SNR}(t)^{-1}$

## What each target finds hard
| Target | High noise (t → T) | Low noise (t → 0) | Notes |
|---|---|---|---|
| **ε** | Easy to output (≈ the input), but the implied x₀ is unstable | **Hard:** must find a tiny noise residue → this is where fine, high-frequency detail is learned | Simon: "low noise challenging → helps generate low-level frequencies" ❓ verify my reading |
| **x₀** | Predicts the blurry average | Easy (≈ copy the input) | **MDM's choice**: geometric losses work directly on $\hat x_0$ → [[Forward kinematics and geometric losses]] |
| **v** | Mostly −x₀ | Mostly ε | Simon: "always hard, combo of x and ε"; standard in flow matching |

Recent counterpoint: JiT ("Back to Basics: Let Denoising Generative Models Denoise", Li & He, 2025) argues that x-prediction is better when the data is high-dimensional but lies on a low-dimensional manifold, since ε and v are full-dimensional noise. Motion (e.g. 263 × 196 numbers per clip) is a cheap place to test this.

## Why it matters here
- A clean, cheap ablation on a fixed dataset: same network, same budget, target ∈ {ε, x₀, v}, with and without matched loss weighting → [[Idea - Prediction target study]]

## Seen in
- MDM (x₀) · MLD (ε) · SALAD (v) · flow-matching models (v) · JiT (x)
