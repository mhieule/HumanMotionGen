---
tags: [course, source]
from: Simon Giebenhain
---
# Kickoff email (verbatim)

> [!note] Kept as the source of truth. The structured version is in [[Course info]].

How will this course will work out:
You will work in a team, such that you can benefit from each other. Splitting up tasks in an efficient manner is a crucial skill to learn. Most likely, you will not directly work on the exact same thing, but rather you will work on different aspects of the same project (see below).

Communication: I will set up a slack space for communication, you will recieve an invitation email shortly.

Weekly Meetings: During the semester we will have weekly online meetings, where you should present your progress, questions, plans for the next weeks, things you are struggling with or which might just not work out as we planned. Please prepare a short presentation every week, it does not have to look super polished, it should just represent your thoughts, and have the most recent experimental results and visualizations.
This environment is similar to most research jobs or start-ups which work on cutting edge problems with lots of uncertainty. Even your supervisor (in this semester that's me) will not be able to tell you the ideal solution or to what degree the project will work as planned. The best your supervisor will be able to give you is a good intuition on what things to try, good background knowledge about which technical solutions are available for smaller subproblems, and a sharp mind that can analyze your experiments and judge what we need to try next to verify our hypothesis. Importantly, the feedback I can give you will be limited by your presentation of your results. Usually, this is quite exciting, and at times a bit frustrating. But ultimately, it is a great opportunity to understand how to work on unsolved problems, e.g. on problems which LLMs cannot solve on their own.

Meeting time: The meeting will probably be ~1h per week, since we are just 2 students (at max). I created a poll in slack where you can pick your preferences. We will do the meetings via Google Meet.

Deadlines: My suggestions would be to have the final presentation in the last week of the semester 01.02.27-05.02.27. Depending on your exam schedule, we can also shift it further back. There will also be a final report due, which will be towards the end of the semester holidays, here I would suggest the 01.03.27-05.03.27. The final presentation will be in person, and I will organize a room.

Workload: The course yields 10 Credits which roughly equates to 300 hours of work. If you split that up evenly throught the semester and round a bit, you can assume two full working days per week, roughly speaking. The expectations are obviously adjusted to the number of people.

GPU access: Unfortunately, you will not have access to our Lab's compute cluster. Instead you will each get a remote PC, which you can access via SSH. I am in contact with our system administrator to set up two machines for you. Note however, that the GPUs are rather small, 10-12GB VRAM.

Project Topic:
@Minh Hieu Le and me were briefly discussing our joint interests. I think Human Motion Generation, could be a great playground to explore modern generative paradigms, like diffusion, flow-matching and auto-regressive models. Furthermore, ther is a connection to robotics, which you are both interested in. Especially Humanoid Robots seem to be one of the hottest topics for the coming decade. And human motion has lots of data sources, so generating human motion to fulfill a task, and then mapping the motion to a robot is one of the most promising ideas in modern 3d computer vision. Furthermore, generating human motion only requires to predict a limited number of parameters, i.e. compared to video generation for example the number of parameters to predict does not explode. It is the perfect balance for this course: a challenging task, lots of things to explore, and feasible compute requirements.
There are also several things which can be explored independently of each other, some sub-problems are:
- dataset preparation and first get an overview of what data is available:
    - human motion
    - human-motion-text-pairs
    - human-motion-3d-environment-pairs
    - human/hand-object interaction
- Exploring different generation paradigms. this can be excellently split up between you:
    - Diffusion models for generation
    - FlowMatching models for generation
    - Auto-Regressice models for generation
- You will soon understand that the representation in which generation happens is crucial, e.g. what do you generate:
    - joint angles is the simplest
    - latent vectors, which describe a human body pose, this need training first, but might yield a smoother latent distribution than joint angles
    - tokens, instead of 1 global latent, you could explore options to tokenize a human body pose into smaller chunks, especially important for auto-regressive models
    - you can also treat it as a image/video generation problem: e.g. predict the 3d poistions in UV-space
- Exploring different types of conditioning mechanisms:
    - text
    - 3d scene
    - goal, e.g. put this object there
    - here you can also get creative

What should you do until next week:
Ask me questions if anything is unclear.
Dig into the literature, what I wrote above already has a lot of pointers, but some of it is already advanced.
The core concepts that you will need to understand:
- Diffusion, Flow-Matching (Those are similar), Auto-Regressive models (just like LLMs)
- What conditional generation is, what types exist in the literature
Getting an overview over the available data is also important.

What are the goals for the first meeting next week:
You should present your own understanding of the core concepts, you should present the most promosing/interesting datasets, and you should present some ideas in which directions you want to explore things. Depending on the day we choose, e.g. Monday, you will not have that much time. In that case a slightly incomplete presentation is fine and we will have the remaining stuff next week.
You dont need to code anything yet, you can, of course, do some dataset exploration etc. but first we focus on understanding.
After your presentation we will try to work out a project plan for the following 1-2 weeks, including in-depth research tasks, dataset research, data preparation, and first experimental steps we need to do on the way to our bigger goal.
