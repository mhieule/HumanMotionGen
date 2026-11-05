# MDM on HumanML3D with an RTX 3060

Oct 31, 2026 · @Minh Hieu Le

Goal: run the pretrained 50-step MDM on HumanML3D test prompts, compare its motions with the real ones, then reproduce its numbers on the 263-D evaluator. Plan for one evening of setup plus an unattended evaluation run; everything fits in 12 GB.

## Before you start

MDM is a Linux codebase: its conda file pins Linux-only packages and its download scripts are bash. On Windows, use WSL2 (Ubuntu) with a current NVIDIA driver; it sees the GPU directly.

- [ ] Check the card: `nvidia-smi` should show about 12 GB. The laptop RTX 3060 has only 6 GB, which is enough for sampling but tight for evaluation.
- [ ] Free disk: about 15 GB for the environment, dataset, evaluators and checkpoint.
- [ ] Install Miniconda and `ffmpeg` (`sudo apt install ffmpeg`).
- [ ] HumanML3D is built from AMASS: register on the AMASS site and accept its license before using the motion data.

## Step 1: Environment (about 30 min)

The repo's environment pins PyTorch 1.7.1 with CUDA 11.0, which has no kernels for RTX 30-series cards (compute capability 8.6 needs CUDA 11.1 or newer). Create the env as written, then swap PyTorch for a build that supports the 3060.

```bash
git clone https://github.com/GuyTevet/motion-diffusion-model.git
cd motion-diffusion-model
conda env create -f environment.yml
conda activate mdm

# Replace PyTorch 1.7.1/CUDA 11.0 with a build for RTX 30xx (last version for Python 3.7)
conda remove -y --force pytorch torchvision torchaudio cudatoolkit
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
pip install "moviepy==1.0.3"   # the plotting code imports moviepy.editor (removed in moviepy 2)
pip install -U gdown            # the pinned gdown often fails on Google Drive

bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh   # the 263-D evaluator (Guo et al.)
```

Check that the GPU is usable before going on:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If PyTorch 1.13 causes errors in MDM's code, fall back to `torch==1.12.1+cu113` with `torchvision==0.13.1+cu113` (index `https://download.pytorch.org/whl/cu113`). Keep the env's matplotlib 3.1.3; newer versions break the plotting script.

## Step 2: Data (about 20 min)

The MDM README now links a ready-processed HumanML3D folder ("Download HumanML3D", Google Drive), so you can skip rebuilding it from AMASS. Download it in the browser, unzip what it contains, and place it so the repo sees `dataset/HumanML3D/`:

| Path in `dataset/HumanML3D/` | What it holds |
| --- | --- |
| `new_joint_vecs/<id>.npy` | The 263-D features, shape (frames, 263), 20 fps. This is what MDM trains on and the evaluator reads |
| `new_joints/<id>.npy` | Plain joint positions, shape (frames, 22, 3). Easiest for plotting |
| `texts/<id>.txt` | About 3 captions per motion, one per line, as `caption#tokens#start#end` |
| `train.txt`, `val.txt`, `test.txt` | Motion ids per split (ids starting with `M` are mirrored copies) |
| `Mean.npy`, `Std.npy` | Normalization statistics for the 263-D features |

Take a first look at one test motion:

```bash
python -c "
import numpy as np
r='dataset/HumanML3D'; mid=open(f'{r}/test.txt').readline().strip()
print(mid, np.load(f'{r}/new_joint_vecs/{mid}.npy').shape, np.load(f'{r}/new_joints/{mid}.npy').shape)
print(open(f'{r}/texts/{mid}.txt').read())
"
```

You should see something like `(120, 263) (120, 22, 3)` and three captions.

## Step 3: Model (5 min)

Use `humanml-encoder-512-50steps`: the original MDM architecture, sampled with 50 diffusion steps instead of 1000, so about 20 times faster with comparable quality. Download it from the README's model list, unzip it into `./save/`, and store the checkpoint path for the next steps:

```bash
ls save/                                   # the unzipped model folder
CKPT=$(ls save/*50steps*/model*.pt | head -n 1)
echo $CKPT
```

The folder also holds `args.json` (MDM reads the model settings from it) and, per the README, the evaluation log the authors got. Keep that log for comparison in Step 5.

Optional: `humanml_trans_dec_512_bert-50steps` is the newer DistilBERT variant ("improved precision"). Run it as a second baseline once the first one works.

## Step 4: See it in action (about 30 min)

Each run takes seconds per motion with the 50-step model. Results land in a `samples_...` folder next to the checkpoint: `results.npy` (joint positions + captions), one stick-figure `sample##_rep##.mp4` per motion, and grid videos `samples_00_to_02.mp4`.

1. Random test-set prompts, 3 variations each, at their ground-truth lengths:

   ```bash
   python -m sample.generate --model_path $CKPT --num_samples 6 --num_repetitions 3
   ```
2. Your own prompt (up to 9.8 s):

   ```bash
   python -m sample.generate --model_path $CKPT --text_prompt "a person walks forward, turns around and sits down" --motion_length 6 --num_repetitions 3
   ```
3. Generated vs. real for the same captions. Save this as `compare_gt.py` in the repo root. It renders 5 random test motions in blue and writes their captions to a prompt file:

   ```python
   import os, random, numpy as np
   from data_loaders.humanml.utils.plot_script import plot_3d_motion
   from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain

   root, out = 'dataset/HumanML3D', 'gt_vis'
   os.makedirs(out, exist_ok=True)
   ids = [l.strip() for l in open(f'{root}/test.txt') if not l.startswith('M')]  # skip mirrored copies
   random.seed(0)
   prompts = []
   for mid in random.sample(ids, 5):
       joints = np.load(f'{root}/new_joints/{mid}.npy')          # (frames, 22, 3) at 20 fps
       caption = open(f'{root}/texts/{mid}.txt').readline().split('#')[0]
       prompts.append(caption)
       clip = plot_3d_motion(f'{out}/{mid}.mp4', t2m_kinematic_chain, joints,
                             title=caption, dataset='humanml', fps=20, vis_mode='gt')
       clip.duration = len(joints) / 20
       clip.write_videofile(f'{out}/{mid}.mp4', fps=20, logger=None)
       print(mid, len(joints), 'frames |', caption)
   open(f'{out}/prompts.txt', 'w').write('\n'.join(prompts))
   ```

   Then generate from the same captions and open both folders side by side:

   ```bash
   python compare_gt.py
   python -m sample.generate --model_path $CKPT --input_text gt_vis/prompts.txt --motion_length 6 --num_repetitions 3
   ```

   `--input_text` uses one length for all prompts, so lengths will differ from the real clips. Compare content, not timing.

What to look for, which links to the physics checks: feet sliding while planted, the body floating or sinking, jitter, and whether all parts of a multi-step caption happen.

## Step 5: Evaluation on the 263-D evaluator

Run the short mode first to check the pipeline, then the full mode for numbers comparable to papers. Use `tmux` or `nohup` so a closed terminal doesn't kill the run.

```bash
# Pipeline check: 1000 test prompts x 5 replications, no MultiModality
python -m eval.eval_humanml --model_path $CKPT --eval_mode debug

# Paper protocol: 1000 test prompts x 20 replications, reports mean and 95% CI
nohup python -m eval.eval_humanml --model_path $CKPT --eval_mode wo_mm > eval_wo_mm.out 2>&1 &

# Optional: adds MultiModality (100 prompts x 30 samples), 5 replications
python -m eval.eval_humanml --model_path $CKPT --eval_mode mm_short
```

- The script fixes batch size at 32 (needed for R-Precision) and writes a `.log` next to the checkpoint.
- Time: the script's own notes say about 3 h (`debug`) and 12 h (`wo_mm`) for the 1000-step model. With 50 steps, expect very roughly 15–30 min and 1–2 h on a 3060 (my estimate). Check the time stamp after replication 0 and multiply.
- Each replication prints two rows: `ground truth` (real test motions) and `vald` (MDM). The ground-truth row is your sanity check.

| Metric | Real motions (should match your ground-truth row) | MDM 50-step, published | Your run |
| --- | --- | --- | --- |
| R-Precision top 1 / 2 / 3 | 0.511 / 0.703 / 0.797 | 0.455 / 0.645 / 0.749 |  |
| FID (lower is better) | 0.002 | 0.489 |  |
| MM-Dist / Matching score (lower is better) | 2.974 | 3.330 |  |
| Diversity (close to real) | 9.503 | not reported |  |
| MultiModality (`mm_short` only) | n/a | 2.290 |  |

The published 50-step row is from the CMDM paper's table (standard 263-D evaluator). The log shipped in the checkpoint zip is the most direct comparison. Gaps of a few hundredths are normal across machines and seeds; a ground-truth row far from the real-motion values means the dataset or evaluators are set up wrong.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `no kernel image is available` or an sm\_86 compatibility warning | PyTorch 1.7.1/CUDA 11.0 from the env file | Do the PyTorch swap in Step 1 |
| `Cannot retrieve the public link` or quota errors in `prepare/*.sh` | Old gdown or Google Drive rate limits | `pip install -U gdown`, retry later, or download the link in the browser and place the files where the script would |
| `No module named 'moviepy.editor'` | moviepy 2.x installed | `pip install "moviepy==1.0.3"` |
| Video writing fails | ffmpeg missing | `sudo apt install ffmpeg` |
| `Can't find model 'en_core_web_sm'` | spaCy model not downloaded | `python -m spacy download en_core_web_sm` |
| Error at `ax.grid(b=False)` | matplotlib was upgraded | `pip install matplotlib==3.1.3` |
| File-not-found under `dataset/HumanML3D` | Data unzipped one level too deep | The folder must directly contain `new_joint_vecs/`, `texts/`, `test.txt` |
| CUDA out of memory | 6 GB laptop card or another GPU process | Lower `--num_samples` for sampling; close other GPU apps |

## Sources

- [MDM repository README](https://github.com/GuyTevet/motion-diffusion-model): setup, data and model links, sampling and evaluation commands
- [MDM environment.yml](https://github.com/GuyTevet/motion-diffusion-model/blob/main/environment.yml): pinned PyTorch 1.7.1 / CUDA 11.0
- [MDM eval/eval\_humanml.py](https://github.com/GuyTevet/motion-diffusion-model/blob/main/eval/eval_humanml.py): evaluation modes, sample limits, time notes
- [MDM plot\_script.py](https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/utils/plot_script.py): the plotting function used in `compare_gt.py`
- [HumanML3D repository](https://github.com/EricGuo5513/HumanML3D): dataset format
- [CMDM paper, Table 1](https://arxiv.org/abs/2602.22594): published MDM 50-step numbers on the 263-D evaluator
