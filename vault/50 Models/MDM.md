---
tags: [model]
paper: "Human Motion Diffusion Model (Tevet et al., ICLR 2023)"
repo: "https://github.com/GuyTevet/motion-diffusion-model"
paradigm: diffusion
target: x0
representation: "HumanML3D 263-D"
text_encoder: CLIP
fits_12gb: "train: yes (expected)"
status: to-try
---
# MDM: Human Motion Diffusion Model

> [!info] The baseline Simon named. Old (2022), but the cleanest base for adding modern image-diffusion techniques → [[Idea - Modernize MDM]]

## Design
- **Backbone:** Transformer **encoder** over frames (not U-Net); the input is the noisy motion sequence
- **Conditioning:** timestep embedding + CLIP text embedding, **prepended as a token** → [[Conditioning mechanisms]]
- **Target:** predicts **x₀** directly → enables geometric losses → [[Prediction targets (epsilon, x0, v)]], [[Forward kinematics and geometric losses]]
- **Schedule:** cosine, 1000 steps (the repo also has faster 50-step models)
- **CFG:** condition dropped during training; guidance scale ~2.5 at sampling → [[Classifier-free guidance]]
- **Tasks:** text-to-motion, action-to-motion, unconditional, editing (inpainting in time and over body parts)

## Compute (verify in the paper/repo)
- Paper: trained on a single GPU (RTX 2080 Ti class) for about 3 days → should fit a 10–12 GB card

## Running it
- [ ] Clone the repo, create the env (`environment.yml`), download the CLIP/spaCy dependencies
- [ ] Get [[HumanML3D]] (or start with the repo's pretrained checkpoints and the text-only data for sampling)
- [ ] Sample with a pretrained model → render the stick-figure mp4s → `99 Attachments`
- [ ] Evaluate the pretrained checkpoint (FID, R-precision) to check the evaluator setup
- [ ] Train → [[E001 - Reproduce MDM on HumanML3D]]

## What to note while running
- Wall-clock time per 1k steps, peak VRAM, batch size used
- How sampling looks at intermediate timesteps → [[Training loss vs sample quality]]

## Notes
- 
