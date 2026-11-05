---
tags: [concept, body]
---
# SMPL and rotation representations

> [!abstract] Primer (replace with your own words)
> **SMPL** is a parametric body model: shape parameters β (body proportions) and pose parameters θ (a rotation per joint on a kinematic tree of 24 joints) produce a mesh and joint positions. **SMPL-H** adds hands and **SMPL-X** adds hands and face. HumanML3D uses 22 body joints. Rotations can be stored as Euler angles, axis-angle, quaternions, rotation matrices or the **6D representation** (first two columns of the rotation matrix; Zhou et al., 2019), which is continuous and therefore easier for networks to regress.

## In my words


## Rotation formats
| Format | Dim | Continuous for learning? | Notes |
|---|---|---|---|
| Euler angles | 3 | No (gimbal lock, wrap-around) | Avoid as a network output |
| Axis-angle | 3 | No (discontinuity at π) | SMPL / AMASS storage format |
| Quaternion | 4 | No (q and −q are the same rotation) | Needs normalising |
| Rotation matrix | 9 | Yes, but over-parametrised | Needs orthogonalising |
| **6D** | 6 | **Yes** | Gram–Schmidt back to a matrix; HumanML3D's rotation features |

## Local vs. global
- Local: each joint's rotation relative to its parent
- Global: the root's position and heading in the world; often stored as velocities (HumanML3D) to be invariant to where and which way the motion starts

## Robot link
Retargeting maps SMPL motion onto a robot skeleton (e.g. Unitree G1) with different proportions and joint limits, via IK or optimisation (e.g. GMR); a physics-based tracking policy then executes it.

## Related
[[Forward kinematics and geometric losses]] · [[Motion representations]]
