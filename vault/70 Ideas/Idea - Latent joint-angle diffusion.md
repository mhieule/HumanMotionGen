---
tags: [idea]
status: proposed
source: "Simon, Meeting 01"
effort: L
---
# Idea: MDM on joint angles, in a learned latent space

## Hypothesis
Raw joint angles are the simplest, robot-friendly representation, but their distribution is awkward (rotations, a chain of dependencies). Encoding them with a VAE whose latent is regularised to **N(0, I)** and running diffusion **in that latent space**, as in MLD, gives a smoother target distribution and better samples than diffusing the angles directly.

## Steps
1. Baseline: MDM-style diffusion **directly on joint angles** (6D rotations + root), with FK losses
2. Train a VAE on joint angles (reconstruction on angles + FK position loss + KL); check reconstruction FID
3. Diffusion / flow in the latent space; compare with step 1 and with the 263-D baseline

## Risks
- The VAE caps the quality; the KL weight needs tuning
- Two training stages = slower iteration

## Links
[[Latent motion diffusion]] · [[Motion representations]] · [[SMPL and rotation representations]] · [[Forward kinematics and geometric losses]]
