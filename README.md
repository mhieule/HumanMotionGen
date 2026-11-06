# HumanMotionGen

Practical course project on text-to-motion generation (HumanML3D), starting from MDM.

- **New machine → train MDM:** [SETUP.md](SETUP.md) (`scripts/setup_mdm.sh`, no Docker, works on RTX 50xx)
- **RTX 3060 laptop, Docker:** [MDM_HumanML3D_RTX3060.md](MDM_HumanML3D_RTX3060.md) (`./mdm.sh`)
- **MDM vs. SALAD vs. MotionHiFlow side by side:** `scripts/compare/` (`setup_baselines.sh`, then `compare.sh`)
- **Research notes:** `vault/` (Obsidian)

Large files live outside git: `mdm_assets/` (data, checkpoints, evaluators), `baselines/` (third-party repos and their envs), `comparisons/` (outputs).
