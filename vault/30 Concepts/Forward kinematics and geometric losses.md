---
tags: [concept, losses]
---
# Forward kinematics and geometric losses

> [!abstract] Primer (replace with your own words)
> **Forward kinematics (FK)** turns local joint rotations (each relative to its parent) plus the root transform into global 3D joint positions, by walking down the kinematic tree. FK is differentiable, so a model that outputs angles can be trained with losses on **positions**. This matters because a plain MSE on angles gives **every joint the same gradient**, while an error at the hip moves the whole leg. Simon: "this is handled by FK, hence geometric losses", and we could **re-weight body parts** through FK.

## In my words


## Key equation
Global transform of joint j, walking from the root: $G_j = G_{\text{parent}(j)} \cdot \begin{bmatrix} R_j & o_j \\ 0 & 1 \end{bmatrix}$, where $o_j$ is the bone offset. Position $p_j$ = translation part of $G_j$.

## Geometric losses in MDM (on the predicted $\hat x_0$)
- **Position:** $\lVert FK(\hat x_0) - FK(x_0) \rVert^2$
- **Foot contact:** foot velocity should be ≈0 on frames labelled as contact → less foot skating
- **Velocity:** match frame-to-frame differences → smoother motion
- These only make sense with x₀ prediction (or after converting the output to x₀) → [[Prediction targets (epsilon, x0, v)]]

## Re-weighting body parts
- Per-joint weights in the position loss: e.g. more weight on hands and feet (end effectors) or on the root trajectory
- Per-joint-group losses: arms, legs, torso
- Weights that depend on the timestep (geometric losses only at lower noise?)
→ [[Idea - Body-part weighted FK losses]]

## Seen in
- MDM (Tevet et al., 2023), HY-Motion, FloodDiffusion 2 (FK loss)
