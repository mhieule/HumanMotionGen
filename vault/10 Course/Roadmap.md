---
tags: [course, plan]
updated: 2026-11-03
---
# Roadmap

> [!warning] Draft
> Built from the slide milestones ([[Course info]]) and [[Meeting 01]]. Agree it with Simon and the teammate, then keep it updated after every meeting.

## Phases
| Phase | Weeks | Goal | Done when |
|---|---|---|---|
| **0 · Survey** | 02.11 – 13.11 | Know which code runs on 12 GB and which data exists | [[Model landscape]] and [[Dataset overview]] filled in; MDM sampling runs |
| **1 · Baseline** | 16.11 – 04.12 | Train an existing baseline (MDM) on HumanML3D on the lab PC; set up the evaluation | [[E001 - Reproduce MDM on HumanML3D]] reaches roughly the paper's FID; samples visualised per timestep |
| **2 · Modernise** | 07.12 – 15.01 | Fixed dataset; vary architecture, prediction target, loss, model size | Ablation table with ≥ 4 controlled runs → [[Idea - Modernize MDM]], [[Idea - Prediction target study]] |
| **3 · Representation and conditioning** | 11.01 – 29.01 | Latent joint angles, FK losses; maybe a scene/goal condition or AR | One clear finding beyond the baseline → [[Idea - Latent joint-angle diffusion]], [[Idea - Body-part weighted FK losses]] |
| **Wrap-up** | 25.01 – 05.02 | Final presentation | Talk given (01.–05.02.2027) |
| **Report** | Feb – 05.03 | Final report | Submitted (01.–05.03.2027) |

Christmas break (≈ 21.12 – 06.01): plan long training runs to run over it.

## Weekly plan
| Week (Mon) | Me | Teammate | Meeting note |
|---|---|---|---|
| 02.11 | Model survey; run MDM sampling; dataset criteria | | [[Meeting 02]] |
| 09.11 | | | |
| 16.11 | | | |
| 23.11 | | | |
| 30.11 | | | |
| 07.12 | | | |
| 14.12 | | | |
| 21.12 | *break: long runs* | | |
| 04.01 | | | |
| 11.01 | | | |
| 18.01 | | | |
| 25.01 | Slides | | |
| 01.02 | **Final presentation** | | |

## Work split (to discuss)
Simon's email suggests splitting by **paradigm**, e.g. one of us does diffusion (MDM line) and the other flow matching or AR, on one **shared data and evaluation pipeline**. Write down who owns what so the "fair distribution" criterion is visible.
- Shared: data loading, evaluator, visualisation, GitHub repo
- Me: 
- Teammate: 
