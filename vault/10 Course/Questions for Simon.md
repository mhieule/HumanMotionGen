---
tags: [course, questions]
---
# Questions for Simon

> [!tip] Rule: ask twice rather than assume. Small questions go straight to Slack; bigger ones go on the next [[Meeting 02|meeting]] agenda.

## Open
- [ ] **Work split:** one paradigm each (diffusion vs. flow matching) on a shared pipeline, or a different split?
- [ ] **Dataset:** start on HumanML3D (standard benchmark) and keep it fixed for the ablations? When should we move to scene or interaction data?
- [ ] **Your remark on ε at low noise** ("challenging → helps generate low-level frequencies"): did I understand it correctly? → [[Prediction targets (epsilon, x0, v)]]
- [ ] **Evaluation:** should physical metrics (foot skating, floating, penetration) be part of it from the start, because of the robotics angle?
- [ ] **Lab PC:** when is it ready? Disk space for AMASS and HumanML3D? Is internet access available for pip/conda and Hugging Face?
- [ ] **Big models:** can HY-Motion (needs ≥ 24 GB) be run for inference anywhere, or do we only use distilled or CPU-offloaded versions?
- [ ] **GitHub:** one shared repo (org or personal)? Any conventions?
- [ ] **Final report:** format, template, page limit?
- [ ] **Final presentation:** length; exact date within 01.–05.02?

## Answered
| Question | Answer | When |
|---|---|---|
| Diffusion or AR first? | Diffusion and flow matching in the first weeks; AR later | [[Meeting 01]] |
| Which condition to start with? | **Text** (the easiest); 3D scene later. Without a condition the search space is too large for diffusion | [[Meeting 01]] |
| How math-heavy should the work be? | Mostly not. Use the architectures that work best in practice; train on MDM with the tuning techniques | [[Meeting 01]] |
| What makes a good result? | Modern architectures on a fixed dataset; vary losses, model sizes and architectures to find the promising approaches | [[Meeting 01]] |
