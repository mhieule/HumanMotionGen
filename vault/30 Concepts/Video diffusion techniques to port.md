---
tags: [concept, research]
---
# Video diffusion techniques to port

> [!abstract] Why this note
> Companion to [[Image diffusion techniques to port]]. A motion clip is closer to a **video** than to an image: a sequence of frames, each with its own structure (joints instead of pixels), and temporal coherence is what breaks first. Video models had to solve long sequences, temporal compression, streaming generation and keyframe/trajectory control, which are our problems too. Motion has far fewer numbers per frame than video (Simon's point in the [[Kickoff email]]), so we can test these ideas on 12 GB. [[Idea - Modernize MDM]] stays the plan; this is the second menu.

## In my words


## The mapping
| Video | Motion |
|---|---|
| frame | pose at one time step |
| patch / pixel within a frame | joint (or body part) |
| spatial attention | attention across joints within one frame |
| temporal attention | attention across frames for one joint |
| optical flow | joint velocities (already part of the 263-D features) |
| first frame (image-to-video) | current pose / motion prefix |
| camera trajectory | root trajectory |
| fps, resolution, duration | fps (20 vs. 30), clip length (40–196 frames in HumanML3D) |

## Menu

### Architecture
| Technique | Source | What it changes | Motion version | Cost |
|---|---|---|---|---|
| **Factorised space–time attention** | Video LDM (Blattmann et al., 2023), Latte (Ma et al., 2024) | Alternate attention within a frame and across time instead of one full attention | One token per joint × frame; alternate joint attention and temporal attention (skeleton-aware, as SALAD / MotionHiFlow do in their encoders) → [[Motion representations]] | medium |
| **Full 3D attention + multi-axis RoPE** | CogVideoX (2024), HunyuanVideo (2024), Wan (2025) | One attention over all space-time tokens; RoPE split across the t, h, w axes | RoPE over time plus a joint-axis embedding | low |
| **Causal temporal VAE** | CogVideoX, Wan-VAE | Compress time ~4×; the first frame is encoded on its own, so images and videos share one latent | Causal motion VAE with temporal compression (CMDM does this) → [[Latent motion diffusion]] | medium |
| **Image-to-video inflation** | Align your Latents (Blattmann et al., 2023), AnimateDiff (Guo et al., 2024) | Pretrain on images, then insert temporal layers and train on video | Pretrain a single-pose prior on all AMASS frames, then add temporal layers and train on sequences | medium |
| **Joint image + video training** | CogVideoX, Wan practice | Images are treated as 1-frame videos | Mix single poses / short clips with full sequences in one batch | low |

### Training recipe
| Technique | Source | What it changes | Motion version | Cost |
|---|---|---|---|---|
| **Timestep shift by length** | SD3 resolution shift; HunyuanVideo, Wan | More tokens need more noise, so t is shifted towards noise | Shift by clip length; check whether long clips improve | trivial |
| **Length bucketing / sequence packing** | NaViT (Dehghani et al., 2023), Movie Gen (2024) | Batch similar lengths, or pack several clips into one sequence with a block-diagonal mask; no padding | Replace MDM's pad-to-196 + mask | low |
| **fps / micro-conditioning** | Stable Video Diffusion (2023) | Condition on fps and a "motion amount" value | Condition on fps to mix 20 fps HumanML3D with 30 fps data (BONES-SEED, HY-Motion); a motion-intensity knob | low |
| **Joint appearance–motion objective** | VideoJAM (Chefer et al., ICML 2025) | Model also predicts optical flow; at sampling, its own flow prediction steers generation ("Inner-Guidance") | Extra head for velocities / foot contacts, used as guidance at sampling → [[Forward kinematics and geometric losses]] | low–medium |
| **Dense re-captioning** | CogVideoX, Sora | A VLM rewrites captions to be long and detailed | LLM-rewritten HumanML3D texts, or SnapMoGen's long captions. Caveat: the evaluator was trained on short texts → [[Motion evaluation metrics]] | medium |

### Long and streaming generation (the robot case)
| Technique | Source | What it changes | Motion version | Cost |
|---|---|---|---|---|
| **Diffusion forcing** | Chen et al. (NeurIPS 2024) | Each frame gets its own noise level | Already used by CMDM and FloodDiffusion → [[Autoregressive motion generation]] | medium |
| **Rolling / FIFO diffusion** | Rolling Diffusion (Ruhe et al., ICML 2024), FIFO-Diffusion (Kim et al., NeurIPS 2024) | Noise increases along a sliding window; one clean frame leaves the queue per step | Endless motion from a model trained on ≤196 frames; FIFO needs no retraining | low |
| **FreeNoise** | Qiu et al. (ICLR 2024) | Training-free long video: reuse and shuffle noise, fuse windows | Long motions and prompt changes mid-sequence | low |
| **Self Forcing** | Huang et al. (NeurIPS 2025) | Train the AR model on its own rollouts (with KV cache) and a sequence-level loss, which closes the train–test gap (exposure bias) | Post-train an AR motion model (e.g. CMDM) on its own rollouts; measure drift over 30 s+ | medium–high |
| **Causal few-step distillation** | CausVid (Yin et al., CVPR 2025) | Distil a bidirectional teacher into a causal, few-step student | Turn a good offline model into a real-time streaming one for robot control (the ARDY / TACD direction) | high |

### Control and guidance
| Technique | Source | What it changes | Motion version | Cost |
|---|---|---|---|---|
| **Spatiotemporal skip guidance** | STG (Hyung et al., CVPR 2025) | Guide away from a weakened copy of the model (skipped layers) instead of / besides the unconditional one | Drop-in alternative to CFG → [[Classifier-free guidance]] | trivial |
| **Keyframe / first–last-frame conditioning** | SVD (image-to-video), Wan FLF2V | Known frames + mask concatenated as extra input channels | Train explicitly for prefix, keyframe and end-pose conditioning; a robot must start from its current pose | low |
| **Camera / trajectory control** | CameraCtrl, MotionCtrl (2024) | Separate branch encodes the camera path | Root-trajectory ControlNet (ACMDM does this) → [[Conditioning mechanisms]] | medium |

## Video as a source of motion (the bigger research angle)
Not tricks to port but a different route: use video data or video generators to get motion.
- **Generate a video, then lift it to 3D:** *Animating the Uncaptured* (ICLR 2026) renders a humanoid mesh, animates it with an image-to-video model, fits SMPL to the video and optimises the mesh motion. Open-vocabulary and no mocap; slow, and pose-estimation errors and physics problems carry over.
- **Train a 3D motion model from 2D video:** *VideoMDM* (Jun 2026, preprint) trains MDM-style diffusion with 2D reprojection losses and no 3D ground truth. HumanML3D FID 0.88 vs. 0.54 for 3D-supervised MDM. This turns internet video into training data.
- **Video straight to robot motion:** *ViT-H* (Oct 2026, preprint) is a diffusion model conditioned on video that outputs Unitree G1 motion directly (no separate human reconstruction and retargeting), post-trained with DPO using tracking success. Code promised.
- **Compute:** training a video generator is out of reach. Inference of a small one (e.g. Wan 2.1 1.3B) is reported around 8 GB, so generating a handful of clips is possible.
- **Fit with Simon's plan:** his main direction is a *fixed* motion dataset, so this is a later-semester or teammate direction → [[Research directions]], [[Questions for Simon]]

## Suggested first picks
Cheap and most specific to motion: factorised joint × time attention, timestep shift by length, fps conditioning, a VideoJAM-style velocity head, FIFO diffusion for long sequences, STG.

## How to test
Same as [[Image diffusion techniques to port]]: same data, evaluator and budget; one change per run; ≥ 2 seeds → [[Experiment log]]. For the streaming rows, also measure **drift over long rollouts** (FID on later windows, foot skating over time) and transition quality when the prompt changes.
