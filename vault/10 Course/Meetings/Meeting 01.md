---
tags: [meeting]
date: 
course: Practical Course
with: Simon Giebenhain
---
# Meeting 01: first presentation and project direction

> [!todo] Fill in the date (property above).

## Action items
- [ ] **Plan which code I can run** on the lab PC → [[Model landscape]]
- [ ] **Literature research: open-source models with code.** For each one, note how feasible it is to run, what the paper focuses on (real-time speed, non-generative goal, …) and its hardware needs. Exclude compute-hungry papers. 2025–2026 first → [[Model landscape]]
- [ ] **Run them:** MDM → [[MDM]], [[E001 - Reproduce MDM on HumanML3D]]; HY-Motion → [[HY-Motion]]
- [ ] **Data:** size, public availability, text descriptions (and whether they're any good), 3D scene and its geometry → [[Dataset overview]]
- [ ] Questions → Slack directly → [[Questions for Simon]]

## Decisions
- **Order:** diffusion and flow matching in the first weeks; autoregressive later.
- **Condition:** start with **text** (the easiest); 3D scene is an option later.
- **Style:** not math-heavy. Use the architectures that work best in practice, and train on MDM with the tuning techniques we discussed.
- **Main direction:** apply the last few years of image-diffusion improvements to human motion. The "coolest" setup: fix the dataset, then vary losses, model sizes and architectures to find the promising approaches.

## Notes (structured)
### Course
- Lab PC to be discussed. 70% of the grade is project work. Close supervision.

### Prediction targets → [[Prediction targets (epsilon, x0, v)]]
- **ε:** at low noise it is challenging → helps generate the low-level (fine) frequencies.
- **Velocity:** always hard, since it combines x and ε. Related to flow matching; one can be reformulated as the other → [[Flow matching]]
- All targets give a loss with a similar structure.
- MDM works well but is old and doesn't use new techniques → look at modern image-diffusion papers and pick a Transformer architecture and its parameters from there → [[Image diffusion techniques to port]], [[Idea - Modernize MDM]]

### Representation and latent space → [[Forward kinematics and geometric losses]]
- With raw joint angles, every joint gets the same gradient.
- Forward kinematics takes care of that, hence the geometric losses.
- We can re-weight body parts through the FK → [[Idea - Body-part weighted FK losses]]
- New idea: **MDM on joint angles** → encode the joint angles into a latent space with an N(0, I) prior, then run diffusion in that latent space as in MLD → [[Latent motion diffusion]], [[Idea - Latent joint-angle diffusion]]

### Conditioning → [[Conditioning mechanisms]]
- With a DiT, a condition token, cross-attention or AdaLN are all easy to add.
- Without a condition, the search space is too large and the diffusion model struggles.
- Idea: text → several candidate bodies → render → validate, as in WorldAgents → [[Idea - Render-and-verify agent]]

### Training and evaluation → [[Training loss vs sample quality]]
- Training is a single step; inference is ~20 steps in a row → sampling errors accumulate and can break the sample.
- The training loss is dominated by heavily noised examples.
- Generating samples and computing metrics often tells you more than the training loss.
- When inspecting a model, look at the generated motion at each timestep.

<details><summary>Raw notes (as typed in the meeting)</summary>

- PC lab will be discussed
- 70% weight on the projects work
- Close supervision
- Perspective:
	- epsilon: low noise challenging -> help gennerate low-level frequencies
	- velocity: always hard (combo of x and epsilon)
		- related to flow matching, can be reformulated
	- IDEA: MDM works well but old, not new techniques
		- modern image diffision models/papers, choose Transformer architecture/params
- Latent space:
	- Each joint same gradient
	- is handled in forward kinematics, hence geometric losses
	- We could reassign weight to different body parts with the FK
- Note:
	- Do not need the math heavy stuffs (mostly)
	- Prefer: most successful architecture in practice, train on MDM with the mentioned tuning techniques
		- new idea: MDM joint angles
		- encode joint angles to latent space, angle has N(0, I), diffision in latent space like in MLD
	- Generally: apply image improvements in recent years on the domain (human motion gen)
- Both loss: similar structure
- Diffustion vs Autoregressive
	- Diff: better intuitionly
		- First weeks: Diffusion and Flow Matching
	- AR: also nice but later in the same
- Put conditions on diffusion:
	- DiT, so condition token, cross-att, AdaLN easy
	- Without conditions: search space too large, hard for diffusion to predict
		- Play around with text condition (most easy) or 3D scene
- IDEA: text -> possible bodies -> render -> validate -> same as WorldAgents
- Plan: which code I could run
- Literature research: open source model with code
	- look out for (feasibility of run, focus of the paper (some could focus on real-time speed, non-gen purpose, etc.) -> find relevant ones, modern ones that I could explore)
	- -> then run them
	- -> look at the hardware requirements, exclude papers with tons of compute needed
	- -> high priority: 2025, 2026
	- MDM
	- HY-Motion
	- Watch out for human motion data, how large, how available (public ofc), text description, are texts good, do they have 3D scene (and its geometry)
	- Coolest: modern appoaarch architecture wise, fix the dataset, play w different losses, model sizes, architectures -> to determine the promising approaches
- Questions: Slack directly
	- Ask twice rather than assuming
	- Do not hesitate to ask questions
- In flow matching/diffusion:
	- Training single step, inference: 20 steps in a row
	- -> sampling can accumulates erros, cann break
	- -> training loss will be dominated by input examples with lot of noise
	- -> often: generating and calculate metrics is better/says more than training loss
	- When looking at the model: can look at the generated motion at each timestep

</details>
