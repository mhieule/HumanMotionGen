---
tags: [model]
paper: "HY-Motion 1.0 (Tencent Hunyuan, Dec 2025)"
repo: ""
paradigm: flow-matching
representation: "SMPL-H joint rotations (verify)"
text_encoder: "Qwen3-8B"
fits_12gb: "inference only, with hacks"
status: to-try
---
# HY-Motion 1.0

> [!info] The large-scale, modern reference Simon named. Useful for its **recipe** (data scale, DiT and flow matching, training stages) and as a **quality reference**, not as something we train.

## Design (verify against the report)
- Flow-matching **DiT**; **1B** parameters (Lite: **0.46B**)
- Text encoder: **Qwen3-8B** (an LLM), which is why the VRAM is so large
- Trained on thousands of hours of motion data in stages (large-scale pretraining → high-quality fine-tuning → alignment); 30 fps

## Hardware
- **Official minimum: 26 GB (full) / 24 GB (Lite)**
- Measured peak for Lite at batch 1: ~16.5 GB, mostly the text encoder (TACD paper)
- On 10–12 GB: only with unofficial tricks: text encoder on the CPU or 4-bit quantised; or use the distilled TACD student (~2.6 GB) if the weights are released

## Plan
- [ ] Check the repo for CPU/offload options for the text encoder
- [ ] Try Lite with the text encoder on the CPU (slow but maybe fine for a few demos)
- [ ] Otherwise ask Simon whether a bigger machine exists for inference → [[Questions for Simon]]
- [ ] Take ideas from the report into [[Image diffusion techniques to port]]

## Notes
- 
