---
tags: [concept, evaluation]
---
# Motion evaluation metrics

> [!abstract] Primer (replace with your own words)
> Text-to-motion results are usually measured with a pretrained **text–motion evaluator** (Guo et al., 2022) that embeds motions and texts. **FID** compares the feature distributions of real and generated motions (realism). **R-Precision** and **MM-Dist** check whether a motion matches its text. **Diversity** and **MultiModality** measure variety. Physical metrics (foot skating, floating, penetration) matter most for robots.

## In my words


## Metrics
| Metric | Measures | Better |
|---|---|---|
| FID | Realism (distance between feature distributions) | ↓ |
| R-Precision top-1/2/3 | Text–motion retrieval accuracy within a batch of 32 | ↑ |
| MM-Dist | Distance between text and motion features | ↓ |
| Diversity | Spread across all generations | → close to real data |
| MultiModality | Spread across generations for the same text | ↑ (within reason) |
| Foot skating / floating / penetration | Physical plausibility | ↓ |

> [!warning] Two incompatible evaluators
> The original 263-D evaluator (Guo et al.) and the 67-D "essential features" evaluator from MARDM give FID numbers on **different scales**. Pick **one** for all our ablations and only compare papers that used the same one. → [[Model landscape]]

## Our protocol (proposal)
- Fixed prompt set plus fixed seeds for visual comparison; full test-set metrics for tables
- Run each evaluation several times (e.g. 3–5) and report mean ± std
- Always pair numbers with videos → [[Training loss vs sample quality]]

## Seen in
- Guo et al., CVPR 2022 (HumanML3D + evaluator); MARDM (CVPR 2025, 67-D evaluator)
