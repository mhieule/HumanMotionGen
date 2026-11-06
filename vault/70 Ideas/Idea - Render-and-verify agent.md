---
tags: [idea]
status: parked
source: "Simon, Meeting 01 (WorldAgents analogy)"
effort: M
---
# Idea: Text → candidate motions → render → verify

## Hypothesis
As in WorldAgents (my seminar paper: Director → Generator → Verifier), sample several candidate motions for a prompt, render them, and let a VLM (or the text–motion evaluator) reject the bad ones or pick the best. This raises quality at inference time without retraining, and may also serve as a **data-filtering** tool (slide milestone 5: "data curation and filtering").

## Pieces needed
- A fast renderer (stick figure or SMPL mesh → frames)
- A judge: VLM on rendered frames, or the text–motion retrieval score
- Selection: best-of-N, or rejection with re-prompting

## Open questions
- Can a VLM judge motion from a few frames, or does it need video?
- Cost per prompt on the lab PC?

## Links
[[Conditioning mechanisms]] · [[Motion evaluation metrics]] · WorldAgents notes in StudyVault (`20 Papers/@erkoc2026worldagents`)
