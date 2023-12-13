import os 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from models.model_hmm import Multiple_reference_frames_HMM
import matplotlib as mpl
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2) 

script_path = str(os.path.dirname(__file__))
filename = script_path + '/data/' + 'reach_target'
policy=Multiple_reference_frames_HMM()
policy.load_data(filename)
policy.train()
mpl.rcdefaults()
fig, ax = plt.subplots()
plt.rc('text', usetex=True)
# Add a title using LaTeX font
# ax.set_title( r'$\alpha$' , fontsize=16)
ax.grid(color='gray', linestyle='-', linewidth=1)
# Customize the background color
ax.set_facecolor('white')
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True)
ax = fig.gca()
ax.set_title('Hidden Markov Model', fontsize=18)

policy.find_the_most_probable_sequence()
for demo_index in range(9):
    df, area, dtw, fde, fad=policy.reproduce(demo_index, ax=ax, compute_metrics=True)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
#save figure
fig.savefig(script_path + '/figs/hmm.png', dpi=1200, bbox_inches='tight')


filename = script_path + '/data/' + 'reach_target_new'

demos = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]


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
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True)
for spine in ax.spines.values():
    spine.set_linewidth(2)
for i in range(9):
    # A, b = demos_A_xdx[i][0], demos_b_xdx[i][0]
    A, b =demos_A_xdx_new[i][0], demos_b_xdx_new[i][0]
    start=np.zeros_like(policy.demos_xdx[i][0])
    start=policy.demos_xdx[i][0]+(demos_b_xdx_new[i][0][0]-policy.demos_b_xdx[i][0][0])
    vel_new=A[0][2:4,2:4] @ np.linalg.inv(policy.demos_A_xdx[i][0][0][2:4,2:4]) @ start[2:]
    start[2:]=vel_new
    policy.generalize(A, b, start, ax=ax)

fig.savefig(script_path+ '/figs/hmm_ood.png', dpi=1200, bbox_inches='tight')

plt.show()