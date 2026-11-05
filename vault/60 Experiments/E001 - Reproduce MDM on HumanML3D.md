---
tags: [experiment]
id: E001
date: 
status: planned
model: MDM
dataset: HumanML3D
representation: 263-D
target: x0
baseline: "paper numbers"
commit: 
config: 
gpu_hours: 
fid: 
r_precision_top3: 
---
# E001: Reproduce MDM on HumanML3D

## Question / hypothesis
Can we train MDM on the 10–12 GB lab PC and get close to the paper's numbers? This is the baseline for every later ablation (slide milestone 1: "train an existing baseline from GitHub").

## Setup
- Repo: [[MDM]] · Data: [[HumanML3D]]
- Changes from the default config: (batch size? steps?)
- Hardware: lab PC, GPU = ?, VRAM = ?

## Steps
- [ ] Evaluate the **pretrained** checkpoint first → checks the evaluator and data pipeline
- [ ] Short training run (e.g. 10k steps) → loss curve + samples, VRAM and speed
- [ ] Full run (over a weekend?), with checkpoints evaluated along the way

## Results
| Checkpoint | FID ↓ | R-Prec top-3 ↑ | MM-Dist ↓ | Diversity → | Notes |
|---|---|---|---|---|---|
| Paper | | | | | |
| Pretrained (our eval) | | | | | |
| Ours | | | | | |

## Samples
<!-- Embed videos: same prompts and seeds as the standard visual set; denoising trajectory at t = T, ¾T, ½T, ¼T, 0 -->

## Observations

## Conclusion / next
