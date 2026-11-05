---
tags: [idea]
status: proposed
source: "Simon, Meeting 01"
effort: S-M
---
# Idea: Body-part weighted FK losses

## Hypothesis
With angles, every joint gets the same gradient. Losses on FK positions fix that, and **per-body-part weights** in those losses let us choose what matters: feet (contact, no skating), hands (interaction), the root trajectory (global path). This should improve physical plausibility and fine detail without hurting FID.

## Minimal experiment
- Representation: joint angles (or 263-D with its rotation part + FK)
- Losses: angle MSE + FK position loss with weights per group {root, legs/feet, arms/hands, torso/head}
- Variants: uniform · feet-heavy · hands-heavy · weights that depend on t
- Metrics: FID, R-precision, **foot skating**, and per-joint position error

## Links
[[Forward kinematics and geometric losses]] · [[Motion evaluation metrics]]
