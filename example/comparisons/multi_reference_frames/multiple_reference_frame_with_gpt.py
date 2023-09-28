import os 
import numpy as np
import matplotlib.pyplot as plt
# from pbdlib.utils.jupyter_utils import *
import warnings
from models.model_gpt import Multiple_Reference_Frames_GPT
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2) 

filename = os. getcwd()  + '/data/' + 'reach_target'

policy=Multiple_Reference_Frames_GPT()
policy.load_dataset(filename)
policy.train()
fig, ax = plt.subplots()
ax.grid(color='gray', linestyle='-', linewidth=1)
# Customize the background color
ax.set_facecolor('white')
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
source_index=2
for target_index in range(9):
    df, area, dtw, fde, fad=policy.reproduce(source_index, target_index, ax=ax, compute_metrics=True)

#save figure
fig.savefig('figs/gpt.png', dpi=1200, bbox_inches='tight')

plt.show()

filename = 'reach_target_new'

pbd_path = os. getcwd()  + '/data/'

demos = np.load(pbd_path + filename + '.npy', allow_pickle=True, encoding='latin1')[()]


### Coordinate systems transformation

demos_A_new= demos['A']
demos_b_new = demos['b']

demos_A_xdx_new = [np.kron(np.eye(2), d) for d in demos_A_new]
demos_b_xdx_new = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b_new]
fig, ax = plt.subplots()
ax.grid(color='gray', linestyle='-', linewidth=1)
# Customize the background color
ax.set_facecolor('white')
ax.set_xlim(-80, 60)
ax.set_ylim(-80, 60)
for i in range(9):
    # A, b = demos_A_xdx[i][0], demos_b_xdx[i][0]
    A, b =demos_A_xdx_new[i][0], demos_b_xdx_new[i][0]
    start=np.zeros_like(policy.demos_xdx[i][0])
    start=policy.demos_xdx[i][0]+(demos_b_xdx_new[i][0][0]-policy.demos_b_xdx[i][0][0])
    vel_new=A[0][2:4,2:4] @ np.linalg.inv(policy.demos_A_xdx[i][0][0][2:4,2:4]) @ start[2:]
    start[2:]=vel_new
    policy.generalize(A, b, start, ax=ax)

fig.savefig('figs/hmm_new.png', dpi=1200, bbox_inches='tight')

plt.show()

