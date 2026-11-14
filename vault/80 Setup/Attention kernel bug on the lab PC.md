---
tags: [setup, bug]
date: 2026-11-14
---
# Attention kernel bug on the lab PC

> [!danger] On the lab PC, PyTorch computes attention **wrong, without any error or warning**, for float32 models whose attention heads are larger than 64 dimensions.
> Training such a model goes fine for a few hundred steps and then collapses; sampling and evaluation give slightly wrong numbers.
> **Before training or evaluating any model on this machine, run the check below in that model's environment.**

## What is wrong
- **Machine:** lab PC, RTX 5070 Ti (Blackwell, compute capability 12.0), driver 595, PyTorch 2.7.1 + CUDA 12.8 (`baselines/mdm/.venv`). Other PyTorch versions are untested.
- **Kernel:** PyTorch has several interchangeable kernels for attention (`F.scaled_dot_product_attention`) and picks one automatically. For float32 on the GPU it picks the **memory-efficient** kernel. `nn.MultiheadAttention`, `nn.TransformerEncoder` / `Decoder` and most model code call attention this way.
- **When it is wrong** (head size = model width ÷ number of heads; measured with `scripts/check_attention_kernel.py`):

| dtype | head size ≤ 64 | head size 72–128 | head size 136–256 |
|---|---|---|---|
| float32 | correct | **outputs and gradients wrong** (errors ~1 on outputs of size ~1) | **gradients wrong** (20–70% relative), outputs correct |
| float16, bfloat16 | correct | correct | correct |

  Flash, cuDNN and math kernels were correct in every tested case (flash and cuDNN do not run float32 at all). The error grows with how sharp the attention is, which is why training only breaks after a while.

## What it did to us
- **MDM** (width 512, 4 heads → head size 128, float32): training diverged at step ~600 for every seed; the gradient norm went from 0.4 to over 2000 and the loss from 0.35 back to 1.15. Replaying one step: gradient norm ~4000–7700 on the GPU, 0.32 on the CPU. Details: [[E003 - MDM-50 baseline training run]].
- **MDM sampling and evaluation** were slightly off: released checkpoint test FID 0.567 measured vs. 0.552 correct, APG 0.463 vs. 0.444. Inside the confidence intervals; [[E002 - Inference-time guidance on MDM (Tier 0)]] conclusions hold.
- **Not affected:** our TMR scorer (width 256, 4 heads → 64), CLIP's text encoder (64-dim heads, float16), the Guo et al. T2M evaluator (GRUs, no attention).

## The fix
Put this right after `import torch` in every script that runs a model on the GPU:
```python
torch.backends.cuda.enable_mem_efficient_sdp(False)   # lab PC: memory-efficient attention is wrong in float32 for head size > 64
```
PyTorch then uses the plain math kernel for float32, which is what PyTorch 1.x always did. Cost: none measurable for MDM (5.4 steps/s either way); larger models with long sequences use more memory.
- Our scripts already do it: `experiments/baseline/*`, `experiments/tier0/eval_tier0.py`.
- A third-party module you don't want to edit: `.venv/bin/python <repo>/scripts/fixed_attention.py -m <module> <args>` (e.g. `-m train.train_mdm`, `-m sample.generate`).
- With the fix, MDM training is bit-reproducible: two runs with the same seed give identical weights.

## Checklist for a new model or environment on this machine
1. Run the check in **that model's** environment (under a minute, prints a table per kernel, dtype and head size; exit code 1 = a default kernel is wrong):
   ```bash
   <env>/bin/python -W ignore scripts/check_attention_kernel.py
   ```
2. Work out the model's attention head size(s) and dtype from its config: head size = width ÷ heads, separately for each attention block (text encoder, denoiser, VAE, evaluator). Mixed precision (`autocast`) runs attention in float16/bfloat16; plain training runs float32.
3. If any of its attention runs in float32 with head size > 64, or the check flags a kernel the model uses, add the fix line to every entry point (training, sampling, evaluation), or launch through `scripts/fixed_attention.py`.
4. Confirm on the model itself: one batch through the model on GPU and on CPU with dropout off should agree to ~1e-5 in float32. `experiments/baseline/diag_spike.py analyze` shows how this was done for MDM.
5. Write the result into the model's note (head size, dtype, affected yes/no).
6. Re-run the check after any PyTorch, CUDA or driver update; drop the fix only when the check passes.

| Model | Attention head size, dtype | Affected? | Checked |
|---|---|---|---|
| MDM (trans_enc, CLIP) | 128, float32 | **yes** | 2026-11-14 |
| TMR (our scorer) | 64, float32 | no | 2026-11-14 |
| CLIP ViT-B/32 text encoder | 64, float16 | no | 2026-11-14 |
| Guo et al. T2M evaluator | no attention (GRU) | no | |
| SALAD, MotionHiFlow | ? | not installed on the lab PC yet | |

## Sources
- Our measurements: `scripts/check_attention_kernel.py`, [[E003 - MDM-50 baseline training run]], design doc `docs/superpowers/specs/2026-11-14-mdm-baseline-run-design.md` ("What happened at launch").
- Closest public report: same silent failure of the memory-efficient kernel on a DGX Spark (sm_121, custom build), [NVIDIA developer forum](https://forums.developer.nvidia.com/t/dgx-spark-sm121-silent-sdpa-efficient-attention-corruption-in-a-custom-pytorch-build-diagnostic-chain-standalone-reproducer-workaround/368005). I found no report for the RTX 50xx with stock PyTorch.
