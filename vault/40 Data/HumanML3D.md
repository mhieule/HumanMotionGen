---
tags: [data, dataset]
type: motion-text
size: "14,616 motions / 44,970 texts / ~28.6 h"
access: "build script from AMASS + HumanAct12"
text: yes
scene: no
fps: 20
status: to-download
---
# HumanML3D

> [!info] The standard text-to-motion benchmark (Guo et al., CVPR 2022). Almost every model in [[Model landscape]] reports on it, so it is our **fixed dataset** for the baseline and the ablations.

## Facts (verify)
- 14,616 motions with 44,970 text descriptions (~3 per motion); ~28.6 h
- Built from **AMASS** + HumanAct12; clips are mirrored for augmentation (left ↔ right in the text, too)
- 20 fps, at most 196 frames (~10 s); 22 joints
- Splits: train / val / test (standard files in the repo)
- Representation: 263-D features per frame → [[Motion representations]]
- Evaluator: Guo et al.'s text–motion encoders (263-D FID); the MARDM 67-D evaluator is the alternative → [[Motion evaluation metrics]]

## How to get it
- [ ] Register for and download AMASS (the subsets listed in the HumanML3D repo, SMPL-H G)
- [ ] Download SMPL-H and DMPL body models
- [ ] Run the HumanML3D notebooks/scripts (raw pose processing → motion representation → `new_joint_vecs`, `texts`)
- [ ] Copy into the MDM repo's `dataset/HumanML3D` → [[MDM]]
- [ ] Disk space needed: ? GB (check before the lab PC fills up) → [[Lab PC]]

## Text quality (my notes)
- 

## Known issues
- The 263-D features are redundant (positions, rotations and velocities can disagree) → MARDM's critique
- Captions are short and generic, so fine-grained control is limited
