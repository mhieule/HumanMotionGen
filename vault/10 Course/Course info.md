---
tags: [course]
semester: WS 2026/27
supervisor: Simon Giebenhain
project: "P3 - Human Motion Diffusion"
final_presentation: 2027-02-01
final_report: 2027-03-01
---
# Course info

Course: **Master's Practical Course: 3D Scanning and Spatial Learning** (Simon Giebenhain). Project **P3: Human Motion Diffusion** (MDM, HY-Motion).
Sources: [[Kickoff email]] and the intro slides (`PracticalCourse.pptx`).

## Format
- Team of up to 2 students. We work on **different aspects of the same project**; splitting the work efficiently is part of the skill being learned (and graded).
- **Weekly online meeting**, ~1 h, Google Meet; day chosen by a Slack poll. Every week, bring a short, rough presentation: progress, questions, plans for the next weeks, what you're stuck on, what didn't work, and the **latest results with visualisations**.
- Communication: **Slack**. Collaboration: **GitHub** (share your GitHub username in Slack).
- Simon's feedback can only be as good as how we present our results.

> [!warning] The slides and the email disagree
> The intro slides are a general deck (in-person meetings on Friday, 14:00–16:00, room 02.09.014; summer-semester dates). For this semester, the email wins: **online weekly meetings**, the deadlines below, and a remote PC instead of a lab PC.

## Deadlines
| What | When | Notes |
|---|---|---|
| Final presentation | **01.–05.02.2027** (last week of the semester) | In person; Simon books a room. Can move later depending on exams |
| Final report | **01.–05.03.2027** (end of the semester break) | |

## Workload
10 ECTS ≈ **300 h** ≈ two full working days per week. Expectations scale with team size.

## Compute
- **No access to the lab cluster.** Each student gets a remote PC over **SSH**.
- Small GPU: **10–12 GB VRAM** → only models that train on one small card. See [[Lab PC]] and [[Model landscape]].

## Grading (individual)
| Part | Weight |
|---|---|
| Project work | **70%** |
| Final presentation | 15% |
| Final report | 15% |

What counts during the semester (slides):
- **Planning:** fair split of work packages, steady weekly progress
- **Working:** GitHub activity, project-meeting presentations, engagement in discussions
- **Outcome:** thoroughness and academic quality, stated goals achieved

## Project brief (from the email)
Human motion generation is a playground for modern generative paradigms (diffusion, flow matching, autoregressive). It connects to robotics: generate a motion for a task, then map it onto a humanoid. There are few parameters per frame (compared with video), so compute stays feasible.

Sub-problems that can be explored independently:
- **Data:** human motion · motion–text pairs · motion–3D-environment pairs · human/hand–object interaction → [[Dataset overview]]
- **Paradigms** (easy to split between us): diffusion · flow matching · autoregressive → [[Diffusion models]], [[Flow matching]], [[Autoregressive motion generation]]
- **Representation** ("crucial"): joint angles · latent vectors (need training first, but may give a smoother distribution) · tokens (per body part, important for AR) · 3D positions in UV space (an image/video problem) → [[Motion representations]]
- **Conditioning:** text · 3D scene · goal (e.g. "put this object there") · creative ideas → [[Conditioning mechanisms]]

## Milestones for P3 (slides)
1. Train an existing baseline from GitHub
2. Choose a suitable dataset after thorough research, and plan a user-friendly conditioning mechanism
3. Train models on that dataset
4. Improve training dynamics and generalisation
5. More data curation and filtering strategies

See [[Roadmap]] for how these map onto weeks.
