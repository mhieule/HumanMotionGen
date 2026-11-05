---
tags: [concept, paradigm]
---
# Flow matching

> [!abstract] Primer (replace with your own words)
> Flow matching learns a velocity field that moves samples from noise to data along a chosen path. With the straight-line path (rectified flow), the network predicts the constant velocity between a noise sample and a data sample, trained with a plain MSE. Sampling solves an ODE (Euler or Heun) in ~10–50 steps. With Gaussian noise, it is a diffusion model with a particular schedule and a velocity target. Simon: "related to flow matching, can be reformulated".

## In my words


## Key equation / picture
Convention used here (SD3 / rectified flow): t = 0 is data, t = 1 is noise.
- Path: $x_t = (1-t)\, x_0 + t\, \varepsilon$
- Target: $v = \dfrac{dx_t}{dt} = \varepsilon - x_0$ → a **combination of x and ε**, which is why Simon says velocity is "always hard"
- Loss: $\lVert v_\theta(x_t, t, c) - (\varepsilon - x_0) \rVert^2$
- Sampling (Euler): $x_{t-\Delta} = x_t - \Delta \cdot v_\theta(x_t, t, c)$, from t = 1 down to 0

> [!warning] Conventions differ
> Some papers flip it (t = 0 noise, t = 1 data, $v = x_0 - \varepsilon$). Always check this before comparing code.

## Practical knobs (from image models)
- **Timestep sampling:** logit-normal or mode-sampling puts more training on intermediate t (SD3), instead of uniform t.
- **Prediction target:** v, x₀ or ε can each be converted into the others → [[Prediction targets (epsilon, x0, v)]]
- **Solver and number of steps:** Euler vs. Heun; 10 vs. 20 vs. 50 steps → evaluate with metrics, not the loss.

## Why it matters here
- Most of the 2025–2026 motion models I found are flow matching: MotionHiFlow, ACMDM, MSFlow, CMDM, HY-Motion → [[Model landscape]]
- A natural second paradigm next to MDM's diffusion → a candidate for splitting work with the teammate.

## Seen in
- Lipman et al. 2023 (Flow Matching); Liu et al. 2023 (Rectified Flow); SiT (Ma et al. 2024); SD3 (Esser et al. 2024); HY-Motion 1.0 (2025)
