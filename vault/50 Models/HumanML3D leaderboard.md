---
tags: [models, benchmark]
updated: 2026-11-06
---
# HumanML3D leaderboard (263-D evaluator)

Interactive chart: https://claude.ai/artifact/ToQb26q1xyHT42QoiYX9gq (private until shared). Related: [[Model landscape]], [[E001 - Reproduce MDM on HumanML3D]].

All numbers are the papers' own, with the standard Guo et al. evaluator (the same one we use for MDM). Sorted by FID. Gaps of a few thousandths in FID are within the papers' 95% confidence intervals.

**Takeaway:** no single winner. ScaleMoGen has the lowest FID (0.030); MoGeFlow has the best text match (R-top3 0.874, above real motion's 0.797). MotionHiFlow, LaMP and RAM tie at FID 0.032. Our MDM-50 baseline: FID 0.489 (authors' log; our re-run 0.527).

| #   | Model            | Year | Venue              | Approach                  | R-top1 ↑ | R-top3 ↑ | FID ↓ | MM-Dist ↓ | Number taken from                                                              |
| --- | ---------------- | ---- | ------------------ | ------------------------- | -------- | -------- | ----- | --------- | ------------------------------------------------------------------------------ |
|     | *Real motion*    |      |                    |                           | 0.511    | 0.797    | 0.002 | 2.974     | HumanML3D test set (Guo et al. evaluator)                                      |
| 1   | ScaleMoGen       | 2026 | arXiv              | Autoregressive tokens     | 0.577    | 0.856    | 0.030 | 2.626     | ScaleMoGen Tab. 1 (arXiv 2605.11704)                                           |
| 2   | LaMP             | 2025 | ICLR'25            | Masked tokens             | 0.557    | 0.843    | 0.032 | 2.759     | ReMoMask-2 Tab. 4 (arXiv 2609.08365)                                           |
| 3   | RAM              | 2026 | arXiv              | Diffusion                 | 0.561    | 0.839    | 0.032 | 2.716     | RAM Tab. 1 (arXiv 2601.14788)                                                  |
| 4   | MotionHiFlow     | 2026 | CVPR'26            | Flow matching             | 0.563    | 0.843    | 0.032 | 2.691     | MotionHiFlow Tab. 1 (arXiv 2604.23264)                                         |
| 5   | MoGenTS          | 2024 | NeurIPS'24         | Masked tokens             | 0.529    | 0.812    | 0.033 | 2.867     | MotionHiFlow Tab. 1 (arXiv 2604.23264)                                         |
| 6   | KinMo            | 2025 | arXiv              | Masked tokens             | 0.532    | 0.821    | 0.039 | 2.901     | RAM Tab. 1 (arXiv 2601.14788)                                                  |
| 7   | Light-T2M        | 2025 | AAAI'25            | Other                     | 0.511    | 0.795    | 0.040 | 3.002     | MotionHiFlow Tab. 1 (arXiv 2604.23264)                                         |
| 8   | COME             | 2025 | arXiv              | Diffusion                 | 0.526    | 0.816    | 0.041 | 2.898     | RAM Tab. 1 (arXiv 2601.14788)                                                  |
| 9   | DC-Motion        | 2026 | arXiv              | Masked tokens + diffusion | 0.528    | 0.812    | 0.041 | 2.945     | DC-Motion Tab. I (arXiv 2606.14721)                                            |
| 10  | ReMoMask-2       | 2026 | arXiv              | Masked tokens             | 0.528    | 0.813    | 0.042 | 2.860     | ReMoMask-2 Tab. 4 (arXiv 2609.08365)                                           |
| 11  | RMG              | 2026 | arXiv              | Flow matching             | 0.525    | 0.805    | 0.043 | 2.930     | RMG Tab. 4 (arXiv 2603.15016)                                                  |
| 12  | MoMask           | 2024 | CVPR'24            | Masked tokens             | 0.521    | 0.807    | 0.045 | 2.958     | MotionHiFlow Tab. 1 (arXiv 2604.23264)                                         |
| 13  | MoGeFlow         | 2026 | arXiv              | Flow matching             | 0.589    | 0.874    | 0.048 | 2.600     | MoGeFlow Tab. 1 (arXiv 2606.11656)                                             |
| 14  | FloodDiffusion 2 | 2026 | arXiv              | Diffusion                 | –        | 0.815    | 0.053 | 2.832     | FloodDiffusion 2 Tab. 4 (arXiv 2609.33167)                                     |
| 15  | BAMM             | 2024 | ECCV'24            | Masked tokens             | 0.525    | 0.814    | 0.055 | 2.919     | MotionHiFlow Tab. 1 (arXiv 2604.23264)                                         |
| 16  | MotionLCM v2     | 2024 | arXiv              | Diffusion                 | 0.553    | 0.837    | 0.056 | 2.773     | RAM Tab. 1 (arXiv 2601.14788)                                                  |
| 17  | FloodDiffusion   | 2025 | CVPR'26            | Diffusion                 | –        | 0.810    | 0.057 | 2.887     | FloodDiffusion 2 Tab. 4 (arXiv 2609.33167)                                     |
| 18  | MoMask++         | 2025 | NeurIPS'25         | Masked tokens             | 0.528    | 0.811    | 0.072 | 2.912     | MotionHiFlow Tab. 1 (arXiv 2604.23264)                                         |
| 19  | SALAD            | 2025 | CVPR'25            | Diffusion                 | 0.581    | 0.857    | 0.076 | 2.649     | MotionHiFlow Tab. 1 (arXiv 2604.23264)                                         |
| 20  | MMM              | 2024 | CVPR'24            | Masked tokens             | 0.504    | 0.794    | 0.080 | 3.091     | DC-Motion Tab. I (arXiv 2606.14721)                                            |
| 21  | StableMoFusion   | 2024 | ACM MM'24          | Diffusion                 | 0.553    | 0.841    | 0.098 | –         | RAM Tab. 1 (arXiv 2601.14788)                                                  |
| 22  | ReMoDiffuse      | 2023 | ICCV'23            | Diffusion                 | 0.510    | 0.795    | 0.103 | 2.974     | MotionHiFlow Tab. 1 (arXiv 2604.23264)                                         |
| 23  | T2M-GPT          | 2023 | CVPR'23            | Autoregressive tokens     | 0.492    | 0.775    | 0.141 | 3.121     | MotionHiFlow Tab. 1 (arXiv 2604.23264); T2M-GPT paper reports 0.116            |
| 24  | EnergyMoGen      | 2025 | CVPR'25            | Diffusion                 | 0.526    | 0.815    | 0.176 | 2.931     | MotionHiFlow Tab. 1 (arXiv 2604.23264)                                         |
| 25  | MotionGPT        | 2023 | NeurIPS'23         | LLM-based                 | 0.492    | 0.778    | 0.232 | 3.096     | DC-Motion Tab. I (arXiv 2606.14721)                                            |
| 26  | IRG-MotionLLM    | 2025 | ECCV'26            | LLM-based                 | 0.535    | 0.820    | 0.242 | 2.785     | MoGeFlow Tab. 1 (arXiv 2606.11656)                                             |
| 27  | MLD              | 2023 | CVPR'23            | Diffusion                 | 0.481    | 0.772    | 0.473 | 3.196     | ScaleMoGen Tab. 1 (arXiv 2605.11704)                                           |
| 28  | **MDM-50 steps** | 2024 | checkpoint release | Diffusion                 | 0.455    | 0.749    | 0.489 | 3.326     | Authors' eval log shipped with humanml_enc_512_50steps (our re-run: FID 0.527) |
| 29  | **MDM**          | 2023 | ICLR'23            | Diffusion                 | –        | 0.611    | 0.544 | 5.566     | ScaleMoGen Tab. 1 (arXiv 2605.11704)                                           |
| 30  | MotionDiffuse    | 2022 | TPAMI'24           | Diffusion                 | 0.491    | 0.782    | 0.630 | 3.113     | ScaleMoGen Tab. 1 (arXiv 2605.11704)                                           |
| 31  | T2M              | 2022 | CVPR'22            | Other                     | 0.455    | 0.736    | 1.087 | 3.347     | ScaleMoGen Tab. 1 (arXiv 2605.11704)                                           |
| 32  | TM2T             | 2022 | ECCV'22            | Autoregressive tokens     | 0.424    | 0.729    | 1.501 | 3.467     | ScaleMoGen Tab. 1 (arXiv 2605.11704)                                           |

## Not on this list (different evaluator, not comparable)
- **67-D evaluator (MARDM):** MARDM, ACMDM, MoLingo, MLA-Gen. E.g. MoLingo FID 0.049, ACMDM-XL 0.058 on that scale.
- **MotionStreamer / TMR-style evaluator:** MotionStreamer, Möbius (Nov 2026), HY-Motion. FIDs there are in the 6–35 range.
- T2M-GPT: most recent tables cite FID 0.141; its own paper reports 0.116.
