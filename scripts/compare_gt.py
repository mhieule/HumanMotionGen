import os, random, numpy as np
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain

root, out = 'dataset/HumanML3D', 'gt_vis'
os.makedirs(out, exist_ok=True)
ids = [l.strip() for l in open(f'{root}/test.txt') if not l.startswith('M')]  # skip mirrored copies
random.seed(0)
prompts = []
for mid in random.sample(ids, 5):
    joints = np.load(f'{root}/new_joints/{mid}.npy')          # (frames, 22, 3) at 20 fps
    caption = open(f'{root}/texts/{mid}.txt').readline().split('#')[0]
    prompts.append(caption)
    clip = plot_3d_motion(f'{out}/{mid}.mp4', t2m_kinematic_chain, joints,
                          title=caption, dataset='humanml', fps=20, vis_mode='gt')
    clip.duration = len(joints) / 20
    clip.write_videofile(f'{out}/{mid}.mp4', fps=20, logger=None)
    print(mid, len(joints), 'frames |', caption)
open(f'{out}/prompts.txt', 'w').write('\n'.join(prompts))
