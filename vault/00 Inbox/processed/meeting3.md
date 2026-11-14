Text-motion evaluator could be dependent on the method/model chosen
-> look how good in the paper, compare with human perception

Diversity to make sure model does not collapse

Also important
- How many videos we need for the FID -> find the right setting
- Get feedback that are reliable, but reasonably cost-efficient (so that we could do eval more frequently)

Proxy run:
- see if 100k steps make sense -> maybe tune, save more frequently to see how the model behave
- at best get some evaluations in 

in the beginning: be very efficient, how the model behaves during the steps (also training curve)

Weight n bias

Worth to run once in full, logs in w n b

Tier 0:
- what works best depends on model, so focus on tier 1 and 2 now. 

Look at other fundamental aspects
- Representations (joint angel/speed, grouping timestep into one latent to reduce the complexity)
-> can be intuitive for human but hard for model to learn/produce good gradient
-> allows for good learning
- Text encoder good enough or too course -> good model better training -> train faster

Architecture + Representation + Data (generate our own caption)!!!

Tier 1+2 most important for the course!

Tier 3: use physics as rewards, could be interesting, depends on good model

Joint angle and noise might behave differently, maybe innterval CFG and logit-normal timesteps could lead to weird joints/motion

Time management: setup a script to try out different settings

Tier 1: large impact approaches first (latent repr. most important and trickest, flow matching vs diff first, QK later in case of instabilities) -> 
Allso learning rate is importatn 
-> Try out fundamental stuffs first, then details

Take inspiration from MotionHiFlow

Currently plan good. try out first, then 