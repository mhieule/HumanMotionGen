---
tags: [data, overview]
updated: 2026-11-03
---
# Dataset overview

> [!warning] Sizes are approximate, from memory and earlier notes
> Check each one on its project page before putting it on a slide, then tick ✔ in the last column.

## What Simon wants to know ([[Meeting 01]])
1. **How large?** Hours, number of clips
2. **How available?** Public; registration or licence
3. **Text descriptions?** And are they any good (length, detail, errors)?
4. **3D scene?** And its geometry (mesh, point cloud, scan)?

## Overview
| Dataset | Type | Size (approx.) | Text | Scene geometry | Body / fps | Access | ✔ |
|---|---|---|---|---|---|---|---|
| **AMASS** | Motion only (unified mocap) | 40+ h, 300+ subjects, ~11k motions | – | – | SMPL-H, varies | Registration | |
| **[[HumanML3D]]** | Motion + text | 14.6k motions, 45k texts, ~29 h | Short, crowd-sourced | – | 22 joints, 20 fps | Script on top of AMASS | |
| KIT-ML | Motion + text | 3.9k motions, 6.3k texts, ~11 h | Short | – | 21 joints, 12.5 fps | Public | |
| BABEL | Motion + action labels | ~43 h of AMASS | Frame-level action labels | – | SMPL-H | Registration (AMASS) | |
| Motion-X / Motion-X++ | Whole-body motion + text | ~81k / ~120k clips | Whole-body + face text | – | SMPL-X | Licence request | |
| SnapMoGen | Motion + text | 44 h, 20k clips | **Long (~48 words)** | – | – | Public | |
| BONES-SEED | Studio mocap | 288 h | ? | – | – | Public | |
| HUMANISE | Motion + scene + text | ~19.6k motions in 643 ScanNet scenes | Template-based | ScanNet scans | SMPL-X | Public | |
| TRUMANS | Motion + scene (+ objects) | ~15 h, ~100 scenes | Action labels | Synthetic scenes | SMPL-X | Public | |
| LINGO | Motion + scene + text | ~16 h, ~120 scenes | Language | Indoor scenes | SMPL-X | Public | |
| GRAB | Whole-body grasping | 10 subjects, 51 objects | – | Object meshes | SMPL-X | Registration | |
| BEHAVE | Human–object interaction | 8 subjects, 20 objects | – | Object meshes | SMPL | Public | |
| OMOMO | Human–object interaction | ~10 h, 15 objects | – | Object meshes | SMPL-X | Public | |
| ARCTIC | Bimanual hand–object | 2.1M frames, 11 articulated objects | – | Object meshes | MANO / SMPL-X | Registration | |
| HUMOTO | Human–object interaction | ~735 sequences, 63 objects | Text | Object meshes | – | Public | |
| FrankenStein | Motion + per-body-part text over time | ? | Per body part | – | – | ? | |

## Shortlist (fill in)
| Purpose | Pick | Why |
|---|---|---|
| Baseline + ablations (fixed dataset) | HumanML3D | Standard benchmark, evaluator exists, MDM ready |
| Better text | | |
| Scene conditioning (later) | | |

## Text quality checks (for any text dataset)
- [ ] Read 50 random captions; note length, vocabulary, errors and mismatches
- [ ] Count duplicates or near-duplicates
- [ ] Left/right mistakes (HumanML3D mirrors clips for augmentation)

Template for a dataset note: *Dataset* in `90 Templates`.
