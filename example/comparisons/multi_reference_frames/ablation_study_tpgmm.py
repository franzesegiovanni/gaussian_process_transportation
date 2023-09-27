import os 
import numpy as np
import matplotlib.pyplot as plt
# from pbdlib.utils.jupyter_utils import *
import warnings
from model_tp_gmm import Multiple_reference_frames_TPGMM
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2) 

filename = os. getcwd()  + '/data/' + 'reach_target'
policy=Multiple_reference_frames_TPGMM()
policy.load_data(filename)
policy.train()
# fig, ax = plt.subplots()

policy.reproduce(plot=True, compute_metrics=False)


# plt.show()

filename = 'reach_target_new'

pbd_path = os. getcwd()  + '/data/'

demos = np.load(pbd_path + filename + '.npy', allow_pickle=True, encoding='latin1')[()]

### Coordinate systems transformation

demos_A_new= demos['A']
demos_b_new = demos['b']
fig, ax = plt.subplots()
for i in range(9):
    # A, b = demos_A_xdx[i][0], demos_b_xdx[i][0]
    A, b =demos_A_new[i][0], demos_b_new[i][0]
    start=policy.starting_point_rel[i] + demos_b_new[i][0][0]
    policy.generalize(A, b, start, ax=ax, plot=True, compute_metrics=False)

# fig.savefig('figs/hmm_new.png', dpi=1200, bbox_inches='tight')

plt.show()