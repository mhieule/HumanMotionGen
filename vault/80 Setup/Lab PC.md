---
tags: [setup]
host: 
user: 
gpu: 
vram_gb: 
cuda: 
---
# Lab PC (remote, SSH)

> [!info] One remote PC per student, set up by Simon's sysadmin. **No cluster access.** GPU: 10–12 GB VRAM.

## Access
- [ ] Credentials received · host / user filled in above
- [ ] `~/.ssh/config` on my laptop:
```
Host motionpc
    HostName <host>
    User <user>
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 6006 localhost:6006   # TensorBoard
```
- [ ] VS Code Remote-SSH works
- [ ] Use `tmux` for every long run (it survives disconnects)

## First checks
- [ ] `nvidia-smi` → GPU model, VRAM, driver/CUDA version → fill in the properties above
- [ ] `df -h` → free disk (AMASS + HumanML3D + checkpoints need tens of GB)
- [ ] Internet: pip / conda / Hugging Face / GitHub reachable?
- [ ] Python env manager: conda or uv; one env per repo

## Folder layout (proposal)
```
~/code/<repo>        # git clones (our repo + baselines)
~/data/HumanML3D     # datasets, shared by all repos (symlink in)
~/data/AMASS
~/runs/E###          # checkpoints + logs per experiment
```

## Known problem: attention kernel (found 2026-11-14)
> [!warning] On this GPU (RTX 5070 Ti, PyTorch 2.7.1 + CUDA 12.8) PyTorch's memory-efficient attention kernel returns wrong values and gradients for **float32 attention with head size > 64**.
> Anything using `nn.MultiheadAttention`, `nn.TransformerEncoder` or `F.scaled_dot_product_attention` that way is affected: MDM (head size 128) training diverges after ~600 steps, sampling is slightly off.
> Full write-up and checklist for new models: [[Attention kernel bug on the lab PC]]
- Fix: `torch.backends.cuda.enable_mem_efficient_sdp(False)` at the top of every script (all scripts in `experiments/` have it).
- Stock MDM commands: `.venv/bin/python ../../scripts/fixed_attention.py -m sample.generate ...`
- Test: `<env>/bin/python -W ignore scripts/check_attention_kernel.py` in each model's environment. Re-run it after any PyTorch or driver update.
- Details: [[E003 - MDM-50 baseline training run]]

## Training on 10–12 GB
- **Precompute text embeddings** once (essential for T5-XXL, LLM2Vec or Qwen encoders), so the encoder never sits in GPU memory during training
- Large text encoders on the **CPU at inference** (Kimodo: `TEXT_ENCODER_DEVICE=cpu`)
- Most recipes use batch 64 → start at **32 + 2-step gradient accumulation**, **bf16**, gradient checkpointing if needed
- Log peak memory: `torch.cuda.max_memory_allocated()`

## Logging and visualisation
- [ ] W&B or TensorBoard (decide; W&B makes sharing with Simon easier)
- [ ] Headless rendering: stick-figure mp4 via matplotlib (as MDM does); SMPL mesh rendering via pyrender/EGL if available

## GitHub (graded: "activity on GitHub")
- [ ] Shared repo with the teammate; small, frequent commits; one branch per experiment
- [ ] Every experiment note records its commit hash → [[Experiment log]]
