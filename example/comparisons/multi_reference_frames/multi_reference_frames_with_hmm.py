import os 
import numpy as np
import matplotlib.pyplot as plt
import pbdlib as pbd
import similaritymeasures
# from pbdlib.utils.jupyter_utils import *
import random
import warnings
import pickle
from generate_random_frame_orientation import generate_frame_orientation
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2)
from model_hmm import Multiple_reference_frames_HMM




if __name__ == "__main__":
    filename = os. getcwd()  + '/data/' + 'reach_target'
    policy=Multiple_reference_frames_HMM()
    policy.load_data(filename)
    policy.train()
    fig, ax = plt.subplots()
    for demo_index in range(9):
        df, area, dtw, fde, fad=policy.reproduce(demo_index, ax=ax, plot=True, compute_metrics=True)
    # plt.grid('on')
    # fig.savefig('figs/hmm.png', dpi=300, bbox_inches='tight')


    filename = 'reach_target_new'

    pbd_path = os. getcwd()  + '/data/'

    demos = np.load(pbd_path + filename + '.npy', allow_pickle=True, encoding='latin1')[()]


    ### Coordinate systems transformation
 
    demos_A_new= demos['A']
    demos_b_new = demos['b']
    
    demos_A_xdx_new = [np.kron(np.eye(2), d) for d in demos_A_new]
    demos_b_xdx_new = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b_new]
    fig, ax = plt.subplots()
    for i in range(9):
        # A, b = demos_A_xdx[i][0], demos_b_xdx[i][0]
        A, b =demos_A_xdx_new[i][0], demos_b_xdx_new[i][0]
        start=np.zeros_like(policy.demos_xdx[i][0])
        start=policy.demos_xdx[i][0]+(demos_b_xdx_new[i][0][0]-policy.demos_b_xdx[i][0][0])
        vel_new=A[0][2:4,2:4] @ np.linalg.inv(policy.demos_A_xdx[i][0][0][2:4,2:4]) @ start[2:]
        start[2:]=vel_new
        policy.generalize(A, b, start, ax=ax, plot=True, compute_metrics=False)

    # fig.savefig('figs/hmm_new.png', dpi=1200, bbox_inches='tight')

    plt.show()