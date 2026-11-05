---
tags: [concept, conditioning]
---
# Classifier-free guidance

> [!abstract] Primer (replace with your own words)
> During training, drop the condition at random (~10% of the time) so that one network learns both $p(x \mid c)$ and $p(x)$. At sampling time, extrapolate from the unconditional prediction towards the conditional one with a guidance scale w. A higher w follows the text more closely (better R-precision) but gives less diverse and sometimes less natural motion.

## In my words


## Key equation
$\hat f = f_\theta(x_t, t, \varnothing) + w \,\big(f_\theta(x_t, t, c) - f_\theta(x_t, t, \varnothing)\big)$
- Costs two forward passes per step (or one with batch size ×2).
- MDM samples with a guidance scale of about 2.5.

## Knobs worth testing
- Guidance scale w (plot FID vs. R-precision)
- Condition-dropout rate during training
- **Guidance only in a limited noise interval** (Kynkäänniemi et al., 2024): skip it at the highest noise levels → [[Image diffusion techniques to port]]

## Seen in
- Ho & Salimans 2022; MDM; almost every text-to-motion model
