---
tags: [experiments, index]
---
# Experiment log

![[Experiments.base]]

## Conventions
- **One note per experiment**, named `E### - short name` (template *Experiment*). A sweep can be one note with a results table.
- **Status:** `planned` · `running` · `done` · `failed` · `abandoned`
- Record the **git commit** and the **config file**, so every row can be reproduced.
- **Change one thing per run** against a named reference run (`baseline:` property).
- Put videos and plots in `99 Attachments/E###/` and embed the best ones in the note.

## Standard evaluation (keep it fixed)
- Dataset and split: HumanML3D test → [[HumanML3D]]
- Evaluator: ☐ 263-D (Guo) ☐ 67-D (MARDM). **Decide once** → [[Motion evaluation metrics]]
- Sampling: steps = 20 (also 50 for the final numbers); CFG scale = ? ; EMA weights
- Visual set: 8 fixed prompts × 3 seeds, plus the denoising trajectory for 2 prompts → [[Training loss vs sample quality]]

## Results table (copy the headline numbers here)
| ID | Change vs. baseline | FID ↓ | R-Prec top-3 ↑ | MM-Dist ↓ | Diversity → | Steps | Verdict |
|---|---|---|---|---|---|---|---|
| E001 | MDM reproduction | | | | | | |
| E002 | Sampler only: APG scale 4, η=0 (vs. CFG 2.5) | 0.463 (base 0.567) | 0.753 (0.750) | 3.310 (3.338) | 9.70 (9.84) | 50 | ✅ free −18% FID → new default sampler |
| E002 | Sampler only: APG + best-of-4 (TMR verifier) | 0.374 | 0.805 | 2.977 | 9.77 | 50 ×4 | ⚠️ 4.6× cost; R@3 above real data → verifier/evaluator overlap |
| E002 | Sampler only: interval CFG (steps 8–49) | 0.593 | 0.752 | 3.343 | 9.84 | 50 | ❌ no gain |
