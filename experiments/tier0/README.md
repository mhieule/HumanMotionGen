# Tier 0: inference-time guidance for MDM

Sampling-only improvements on a trained MDM: no retraining, no new weights. Results: [RESULTS.md](RESULTS.md).

| Method | What it changes | Cost vs. MDM's CFG |
|---|---|---|
| CFG scale sweep | guidance weight (MDM default 2.5) | same |
| Interval CFG (Kynkäänniemi et al. 2024) | guidance only on steps `t_lo..t_hi`; other steps run the conditional pass only | fewer network passes |
| APG (Sadat et al. 2024) | down-weights the part of the guidance update parallel to the conditional x₀ prediction (`eta`), optional negative momentum and norm threshold | same |
| Best-of-N | N candidates per prompt, keep the best by TMR text-motion similarity (`tmr`) or TMR + physics rank sum (`tmr_phys`) | ~N× |

The verifier never uses the T2M evaluator that computes FID / R-precision.

## Files
- `guidance.py`: `GuidedModel` (CFG / interval / APG, caches the CLIP text embedding), `BestOfNDiffusion`
- `scoring.py`: TMR scorer (own re-implementation of TMR's encoder, loads only its weights), physics stats, verifiers
- `eval_tier0.py`: evaluates one config (MDM's metrics + TMR sim + physics + time + NFE), writes JSON
- `pipeline.py`: val sweep → selection → test comparison → `RESULTS.md`; resumable; `--quick` for a 5-minute test
- `checks.py`: sanity checks (wrapper matches MDM's CFG, APG(eta=1) = CFG, TMR retrieval on real motions)

## Run
Needs the setup from [SETUP.md](../../SETUP.md), plus TMR (`baselines/TMR`, weights from its
`prepare/download_pretrain_models.sh`, unpacked to `baselines/TMR/models/`) and
`uv pip install --python baselines/mdm/.venv "transformers==4.46.3" einops "numpy==1.23.5"`.

```bash
cd baselines/mdm
tmux new -s tier0 '.venv/bin/python ../../experiments/tier0/pipeline.py 2>&1 | tee -a save/tier0/pipeline.log'
# one config by hand:
.venv/bin/python ../../experiments/tier0/eval_tier0.py --model_path save/humanml_enc_512_50steps/model000750000.pt \
    --name apg_s6 --split val --reps 5 --scale 6 --apg_eta 0 --apg_momentum -0.5
```
Outputs: `save/tier0/{val,test}/<config>.{json,log,stderr}` (= `mdm_assets/save/tier0/`), `save/tier0/pipeline.log`.
On the RTX 5070 Ti: ~100 s per replication of 1000 motions for 1-candidate configs; best-of-4 ~4.5×.
