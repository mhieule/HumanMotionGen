---
tags: [concept, representation, paradigm]
---
# Latent motion diffusion

> [!abstract] Primer (replace with your own words)
> First train an autoencoder (usually a VAE) that compresses a motion into a small latent. Then train the diffusion or flow model **in that latent space** and decode at the end. MLD (Chen et al., CVPR 2023) showed this is much faster than diffusion on raw motion. Simon's idea: encode **joint angles** into a latent regularised towards **N(0, I)**, then run diffusion there, so the distribution may be smoother and better behaved than raw angles.

## In my words


## Design choices
- **Latent shape:** one global vector per clip (MLD) vs. a latent per time step (SALAD, MotionHiFlow, CMDM) vs. per body part
- **Regularisation:** KL weight (too high → blurry reconstructions; too low → irregular latent); normalise the latents before diffusion (as Stable Diffusion's scale factor does)
- **Skeleton-aware encoder:** graph or per-joint convolutions (SALAD, MotionHiFlow)
- **Decoder losses:** reconstruction on angles **plus** FK position losses → [[Forward kinematics and geometric losses]]

## Risks
- VAE reconstruction quality caps the final quality → measure reconstruction FID first
- Two training stages = two sets of hyperparameters

## Seen in
- MLD (2023), SALAD (CVPR 2025), MotionHiFlow (CVPR 2026), CMDM (CVPR 2026) → [[Model landscape]]
- Idea note: [[Idea - Latent joint-angle diffusion]]
