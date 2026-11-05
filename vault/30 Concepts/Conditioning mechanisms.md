---
tags: [concept, conditioning]
---
# Conditioning mechanisms

> [!abstract] Primer (replace with your own words)
> Conditional generation learns $p(x \mid c)$. The condition is embedded (a text encoder, a scene encoder, …) and injected into the denoiser. Simon: with a DiT-style Transformer, a **condition token**, **cross-attention** and **AdaLN** are all easy to add. Without a condition, the search space is too large and diffusion struggles, so start with **text** (the easiest), then perhaps a 3D scene.

## In my words


## How to inject the condition
| Mechanism | How | Good for | Example |
|---|---|---|---|
| **Condition token** (prefix) | Prepend the embedding as an extra sequence token | One global vector (a sentence embedding) | MDM: CLIP token + timestep token |
| **Cross-attention** | Motion tokens attend to condition tokens | Many tokens (per-word text, scene points) | MLD-style, many T2M models |
| **AdaLN / AdaLN-Zero** | Condition → scale/shift (+ gate) in each LayerNorm | Global signals (timestep, pooled text) | DiT; standard in modern image models |
| **Concatenation / inpainting** | Fix known frames or joints during sampling | Keyframes, trajectories | MDM editing, ACMDM |
| **ControlNet branch** | Trainable copy of the encoder fed the control signal | Spatial control (joint paths) | OmniControl, ACMDM |
| **Guidance at sampling time** | Gradient of a cost (e.g. reach a goal) | Goals, collisions | GMD, scene-aware methods |

## Kinds of condition
- **Text:** CLIP (MDM), DistilBERT, T5, LLM encoders (Qwen3 in HY-Motion). Big encoders → precompute the embeddings ([[Lab PC]]).
- **3D scene:** point cloud, occupancy grid, scene-object features → [[Idea - Scene-conditioned motion]]
- **Goal:** "put this object there", target positions
- **Past motion, keyframes, audio/music**
- Strength vs. diversity is tuned with [[Classifier-free guidance]]

## Seen in
- MDM, MLD, DiT (Peebles & Xie 2023), OmniControl, HUMANISE, TRUMANS
