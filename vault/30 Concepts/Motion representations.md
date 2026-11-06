---
tags: [concept, representation]
---
# Motion representations

> [!abstract] Primer (replace with your own words)
> **What** the model generates matters a lot (Simon: "crucial"). The same motion can be stored as joint angles, 3D joint positions, a redundant feature vector (HumanML3D's 263-D), a learned latent, discrete tokens, or even an image (positions in UV space). Each choice changes how smooth the data distribution is, which losses are possible, and whether the output can drive a body or robot directly.

## In my words


## Options
| Representation | + | − | Example |
|---|---|---|---|
| **Joint angles** (rotations per joint + root) | Simplest; drives SMPL or a robot directly; no bone-length drift | Every joint gets the same loss gradient although parent errors move the whole chain → needs FK losses | MDM-on-angles idea, HY-Motion (SMPL-H rotations) |
| **HumanML3D 263-D features** | The standard benchmark; evaluators exist | Redundant (positions + rotations + velocities can disagree); MARDM argues it hurts diffusion | MDM, MLD, MoMask |
| **Absolute joint positions** | Simple, matches the metrics, easy control | Needs IK for rotations; bone lengths can drift | ACMDM, MSFlow |
| **Latent vectors** (VAE) | Smoother, compact distribution, fast | Must train a VAE first; reconstruction caps quality | MLD, SALAD, MotionHiFlow → [[Latent motion diffusion]] |
| **Tokens** (VQ, per body part) | Fits AR / LLM machinery | Quantisation error; codebook collapse | T2M-GPT, MoMask → [[Autoregressive motion generation]] |
| **3D positions in UV space** | Reuse image/video diffusion backbones | Unusual; mapping overhead | Simon's suggestion |

## HumanML3D 263-D layout (per frame, 22 joints)
root angular velocity (1) · root linear velocity in the xz-plane (2) · root height (1) · local joint positions (21×3) · joint rotations in 6D (21×6) · joint velocities (22×3) · foot contacts (4). At 20 fps, up to 196 frames. → [[HumanML3D]]

## Related
- [[SMPL and rotation representations]] · [[Forward kinematics and geometric losses]]
