---
tags: [home]
---
# Human Motion Generation: Practical Course WS 2026/27

> [!info] At a glance
> **Supervisor:** Simon Giebenhain · **Team:** me + teammate (TBD) · **Credits:** 10 ECTS ≈ 300 h ≈ 2 working days per week
> **Weekly meeting:** ~1 h on Google Meet (day set by the Slack poll) · **Questions:** Slack, anytime
> **Final presentation:** 01.–05.02.2027, in person · **Final report:** 01.–05.03.2027
> **GPU:** remote PC over SSH, 10–12 GB VRAM → [[Lab PC]]
> **Grading:** project work 70% · final presentation 15% · final report 15% (individual) → [[Course info]]

## Now
- **Phase 0: survey.** Which code can I run, which data exists → [[Roadmap]]
- Next meeting: [[Meeting 02]] · last meeting: [[Meeting 01]]
- Open questions: [[Questions for Simon]]
- Top ideas: [[Research directions]]

## Map
| Folder | What lives there | Start here |
|---|---|---|
| `00 Inbox` | Quick captures. New notes land here; sort them weekly | |
| `10 Course` | Organisation, plan, meetings, deliverables | [[Course info]], [[Roadmap]], [[Deliverables]] |
| `20 Papers` | One note per paper (`@citekey`), imported from Zotero | ![[Papers.base]] |
| `30 Concepts` | One idea per note, in your own words | see list below |
| `40 Data` | Datasets: size, access, text quality, scene geometry | [[Dataset overview]] |
| `50 Models` | Code we can run: repo cards, hardware needs | [[Model landscape]] |
| `60 Experiments` | Lab notebook: one note per experiment | [[Experiment log]] |
| `70 Ideas` | Research directions, each with a minimal experiment | [[Research directions]] |
| `80 Setup` | Lab PC, environment, tooling | [[Lab PC]] |
| `90 Templates` | Meeting, Experiment, Concept, Dataset, Model, Idea, Zotero paper | |
| `99 Attachments` | Images, videos, Zotero snapshots | |

## Concepts
- **Generative paradigms:** [[Diffusion models]] · [[Flow matching]] · [[Prediction targets (epsilon, x0, v)]] · [[Autoregressive motion generation]] · [[Latent motion diffusion]]
- **Conditioning:** [[Conditioning mechanisms]] · [[Classifier-free guidance]]
- **The body:** [[SMPL and rotation representations]] · [[Forward kinematics and geometric losses]] · [[Motion representations]]
- **Doing it well:** [[Motion evaluation metrics]] · [[Training loss vs sample quality]] · [[Image diffusion techniques to port]] · [[Video diffusion techniques to port]]

## Simon's rules of thumb
- Ask twice rather than assume. Slack is fine for small questions.
- Judge a model by **generated samples and metrics**, not by the training loss → [[Training loss vs sample quality]]
- Prefer what works in practice over math-heavy methods.
- Literature: open-source code, 2025–2026 first, exclude anything that needs a lot of compute.
- Main direction: take recent image-diffusion improvements and apply them to human motion, on a **fixed dataset** → [[Image diffusion techniques to port]]

## Weekly loop
1. **During the week:** one note per run in `60 Experiments` (template *Experiment*). Save videos and plots to `99 Attachments`.
2. **Day before the meeting:** new note in `10 Course/Meetings` (template *Meeting*). Make 4–6 rough slides: progress, latest results with visualisations, questions, struggles, plan.
3. **After the meeting:** turn action items into tasks. Update [[Roadmap]] and [[Questions for Simon]].
4. **Weekly review (15 min):** empty `00 Inbox`; put new insights into `30 Concepts`; push the code to GitHub (GitHub activity is graded).

## One-time setup
- [ ] Obsidian → *Open folder as vault* → `OneDrive - TUM/TUM/WS2627/Practical Course`
- [ ] In File Explorer, right-click `Practical Course` → **Always keep on this device**
- [ ] Settings → Community plugins → turn on **Zotero Integration** and **Excalidraw** (copied from StudyVault with the same settings: papers go to `20 Papers/@{{citekey}}.md`)
- [ ] Zotero: make a collection `Practical Course` and tag its items `practical`
- [ ] Tables use the built-in **Bases** plugin (`.base` files), so Dataview is not needed
