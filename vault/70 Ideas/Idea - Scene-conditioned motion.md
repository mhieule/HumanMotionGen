---
tags: [idea]
status: parked
source: "Kickoff email, Meeting 01"
effort: L
---
# Idea: Scene- or goal-conditioned motion

## Hypothesis
Robots act in 3D environments, so conditioning on the scene (geometry) or a goal ("walk to the chair and sit") is the step toward the robotics application. Simon: text is the easiest start; a 3D scene is an option for later.

## Options
- Scene encoder (point cloud / occupancy grid around the body) → cross-attention tokens
- Goal as keyframe or target position → inpainting / ControlNet / guidance
- Data: HUMANISE, TRUMANS, LINGO → [[Dataset overview]]

## Links
[[Conditioning mechanisms]] · [[Classifier-free guidance]]
