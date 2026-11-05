---
tags: [concept, evaluation, training]
---
# Training loss vs sample quality

> [!abstract] Primer (Simon, [[Meeting 01]])
> In diffusion and flow matching, **training is a single step** (denoise once from a random t), but **inference is ~20 steps in a row**. Errors accumulate during sampling and can break the result, even when the per-step loss looks fine. The training loss is also **dominated by the heavily noised examples**, where the error is large no matter how good the model is. So **generating samples and computing metrics says more than the training loss**, and it helps to look at the generated motion **at each timestep**.

## In my words


## What to do in practice
- [ ] Every N steps: sample a fixed set of ~8 prompts with fixed seeds → videos in `99 Attachments`
- [ ] Every M steps: FID / R-precision on a subset of the test set → [[Motion evaluation metrics]]
- [ ] **Denoising-trajectory view:** render the predicted $\hat x_0$ at t = T, ¾T, ½T, ¼T, 0. Shows whether the coarse motion is right early and whether the details break late
- [ ] Log the loss **per noise-level bucket** (e.g. t in 5 bins) instead of only the average
- [ ] Keep EMA weights and evaluate them, not only the raw weights
- [ ] Compare the number of sampling steps (10/20/50) and the solver at a fixed checkpoint

## Why it matters here
Every experiment note has a "Samples" section for this reason → [[Experiment log]]
