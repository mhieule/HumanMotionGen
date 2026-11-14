
HY: look at the interesting stuffs, scale down, maybe Sparse Attention to try to fit
have to take care of training speed
-> try to find the good loss
-> organize training well, training architectures,
use validate set held-out, focus on training time and suitable loss/evaluation metrics -> what are the papers using

-> Needs to nail down what are the metrics used to compare against baseline. Is it feasible for us to compute? 

x-axis: training time
y-axis: performance

-> decide accordingly
-> Also judge the efficiency: training time and loss (there could be good losses but expensive to compute)
-> we want a quantifiable way to judge the efficiency of methods 

other important problem: eval metrics

companies:
Start with a small model family -> fast experiment -> try to develop some scaling laws
same for us: try to begin with smaller models, training more times a week, then estimate the scaling laws, then at the end of the sem: model with best effort

scaling law does not need to be estimated too precisely, important is the fair way to judge: val set, val metrics, fair dataset do not leak into another training

--------------------------------------

CLIP is a bit too compressed (only one vector)
-> we could look at the text descriptions in the dataset, judge if CLIP is too simple for the encoding
-> maybe we could gauge the effect of change of encoder into the performance


-------------------------------------
The diffusion architecucture:
-> trend starts from image/video gen, then applied to the niche like human motion gen

-------------------------------------
First try to train things from scratch, then finetune later if it makes sense

-----------------------
Inference-only: we could say we ignore these models and focus only on models that we could train -> we could learn more important details

MotionHiFlow: hierarchy makes sense + latent (chunk hundert frame sequence into 20 chunks, each chunk one latent) -> have an impact on the training time/compute

SALAD: great to start because less tricks, MoitionHiFlow onnly if we could explain the hierarchy

Start with Motion+text, move to Motion+scene+text later (motion maybe not too fast/accurate)
SnapMoGen: similar w HumanML3D but longer texts (can gauge the encoder)

How often does a prompt appear in a dataset
Prompts: needs to have a challenging degree -> how to quantify the challenge: as complicated as the most 5% difficult prompts from the dataset
-> each 5 steps: plot and check but we need to quantify in the long term

FID difficult to compute: how many samples do we need (research) from the GT and the output -> check out literature
--> alternatives -> check out literature


Until next week:
- Training run!
- FID research, how expensive


Look at the new papers for new metrics

VLM for evaluation only
Loss:
_ closed: hard to differentiate, only RL
_ open: expensive, could try later


Q1 high-level: very expensive, not usable for the course. Also physical constraints are not hold. Maybe could use for data generation.



ACTIONs:
Training of MDM, SALAD, Himotion
- Where we are
- how long
- which parameters
Datasets: fixed HumanML3D with complicated prompts
Open: metrics/loss, how much examples do we need for FID for examples
Plot side-by-side: train hiflow architecture and salad loss for example, etc. Or they could be very similar then we would not need to change the details so much. Or could be that we could look at the failed cases
Send results to Teams
