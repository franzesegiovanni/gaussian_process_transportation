import os 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from models.model_tp_gmm import Multiple_reference_frames_TPGMM
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2) 

script_path = str(os.path.dirname(__file__))
filename = script_path + '/data/' + 'reach_target'
policy=Multiple_reference_frames_TPGMM()
policy.load_data(filename)
policy.train()
fig, ax = plt.subplots()
ax.grid(color='gray', linestyle='-', linewidth=1)
# Customize the background color
ax.set_facecolor('white')
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_title('TP Gaussian Mixture Model', fontsize=18)
policy.reproduce(ax=ax)
for spine in ax.spines.values():
    spine.set_linewidth(2)
#save figure
fig.savefig(script_path+'/figs/tp_gmm.png', dpi=1200, bbox_inches='tight')

# plt.show()

filename = script_path + '/data/' + 'reach_target_new'

demos = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]

### Coordinate systems transformation

demos_A_new= demos['A']
demos_b_new = demos['b']
fig, ax = plt.subplots()
ax.grid(color='gray', linestyle='-', linewidth=1)
# Customize the background color
ax.set_facecolor('white')
ax.set_xlim(-80, 60)
ax.set_ylim(-80, 60)

for i in range(9):
    # A, b = demos_A_xdx[i][0], demos_b_xdx[i][0]
    A, b =demos_A_new[i][0], demos_b_new[i][0]
    start=policy.starting_point_rel[i] + demos_b_new[i][0][0]
    policy.generalize(A, b, start, ax=ax)
for spine in ax.spines.values():
    spine.set_linewidth(2)
fig.savefig(script_path+'/figs/tp_gmm_new.png', dpi=1200, bbox_inches='tight')

plt.show()

