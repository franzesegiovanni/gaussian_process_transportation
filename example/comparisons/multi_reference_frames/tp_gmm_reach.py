# This code is heavily based on the code from https://github.com/BatyaGG/Task-Parameterized-Gaussian-Mixture-Model
#if you want to run it, you must copy this file and A.npy and B.npy in the original repository.

import numpy as np
from matplotlib import pyplot as plt
from tp_gmm.TPGMM_GMR import TPGMM_GMR
from tp_gmm.sClass import s
from tp_gmm.pClass import p
from copy import deepcopy

import os
import numpy as np
from policy_transportation.utils import resample
nbVar = 3
nbFrames = 2
nbStates = 3
nbData = 100

### Load data
filename = '/data/reach_target'

pbd_path = os. getcwd() 

demos = np.load(pbd_path + filename + '.npy', allow_pickle=True, encoding='latin1')[()]

resample_lenght = nbData
### Trajectory data
demos_x = demos['x'] # position
for i in range(len(demos_x)):
    demos_x[i] = resample(demos_x[i], resample_lenght)
    demos_x[i] = np.hstack([np.linspace(0, 2,resample_lenght).reshape(-1,1), demos_x[i]]) # add the time at the beginning of the vector
### Coordinate systems transformation
# demos_A = [d for d in demos['A']]
# demos_b = [d for d in demos['b']]

demos_A = demos['A']
demos_b = demos['b']

# Preparing the samples----------------------------------------------------------------------------------------------- #
slist = []
starting_point_rel=[]
for i in range(len(demos_A)):
    pmat = np.empty(shape=(2, resample_lenght), dtype=object)
    tempData = demos_x[i].transpose()
    for j in range(2):
        for k in range(resample_lenght):
            A=np.eye(3)
            B=np.zeros((3,1))    
            A[1:,1:]=demos_A[i][0][j]
            B[1:,0]=demos_b[i][0][j].reshape(-1,)
            A[1:,1:]=-np.eye(2) @ A[1:,1:]
            pmat[j, k] = p(A, B, np.linalg.inv(A), 3)
    starting_point_rel.append(tempData[1:,0]- demos_b[i][0][0].reshape(-1,))
    slist.append(s(pmat, tempData, tempData.shape[1], 3))

print_points_frames=np.zeros((len(demos_x),4,2))
print_points_frames_new=np.zeros((len(demos_x),4,2))
final_distance=np.zeros((len(demos_x),2))
final_orientation=np.zeros((len(demos_x),1))
for i in range(len(demos_x)):
    print_points_frames[i,0,:]=demos_b[i][0][0]
    print_points_frames[i,1,:]=demos_b[i][0][0]+demos_A[i][0][0] @ np.array([ 0, 10])
    print_points_frames[i,2,:]=demos_b[i][0][1]
    print_points_frames[i,3,:]=demos_b[i][0][1]+demos_A[i][0][1] @ np.array([ 0, -10])

# Creating instance of TPGMM_GMR-------------------------------------------------------------------------------------- #
TPGMMGMR = TPGMM_GMR(nbStates, nbFrames, nbVar)

# Learning the model-------------------------------------------------------------------------------------------------- #
TPGMMGMR.fit(slist)

# Reproduction for new parameters------------------------------------------------------------------------------------- #
rnewlist=[]
for i in range(len(demos_A)):
    pmat_new = np.empty(shape=(2, resample_lenght), dtype=object)
    for j in range(2):
        for k in range(resample_lenght):
            A=np.eye(3)
            B=np.zeros((3,1))
            A[1:,1:]=demos_A[i][0][j]
            B[1:,0]=demos_b[i][0][j].reshape(-1,)
            A[1:,1:]=-np.eye(2) @ A[1:,1:]
            pmat_new[j, k] = p(A, B, np.linalg.inv(A), 3)
    starting_point_rel_new=starting_point_rel[i]+ demos_b[i][0][0].reshape(-1,)
    rnewlist.append(TPGMMGMR.reproduce(pmat_new, starting_point_rel_new))

#plot the new trajectories
plt.figure()
for i in range(len(demos_A)):
    plt.plot(print_points_frames[i][0:2,0],print_points_frames[i][0:2,1], linewidth=10, alpha=0.9, c='green')
    plt.plot(print_points_frames[i][2:4,0],print_points_frames[i][2:4,1], linewidth=10, alpha=0.9, c=[30.0/256.0,144.0/256.0,255.0/256.0])
    plt.scatter(print_points_frames[i][0,0],print_points_frames[i][0,1], linewidth=10, alpha=0.9, c='green')
    plt.scatter(print_points_frames[i][2,0],print_points_frames[i][2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
    plt.plot(rnewlist[i].Data[1,:], rnewlist[i].Data[2,:], 'b')

filename = '/data/reach_target_new'

pbd_path = os. getcwd()

demos = np.load(pbd_path + filename + '.npy', allow_pickle=True, encoding='latin1')[()]


### Coordinate systems transformation

demos_A_new= demos['A']
demos_b_new = demos['b']
# print(len(demos['A']))

for i in range(len(demos_x)):
    print_points_frames_new[i,0,:]=demos_b_new[i][0][0]
    print_points_frames_new[i,1,:]=demos_b_new[i][0][0]+demos_A_new[i][0][0] @ np.array([ 0, 10])
    print_points_frames_new[i,2,:]=demos_b_new[i][0][1]
    print_points_frames_new[i,3,:]=demos_b_new[i][0][1]+demos_A_new[i][0][1] @ np.array([ 0, -10])

# Reproduction for new parameters------------------------------------------------------------------------------------- #
rnewlist=[]
for i in range(len(demos_A_new)):
    pmat_new = np.empty(shape=(2, resample_lenght), dtype=object)
    for j in range(2):
        for k in range(resample_lenght):
            A=np.eye(3)
            B=np.zeros((3,1))
            A[1:,1:]=demos_A_new[i][0][j]
            B[1:,0]=demos_b_new[i][0][j].reshape(-1,)
            A[1:,1:]=-np.eye(2) @ A[1:,1:]
            pmat_new[j, k] = p(A, B, np.linalg.inv(A), 3)
    starting_point_rel_new=starting_point_rel[i]+ demos_b_new[i][0][0].reshape(-1,)
    print(starting_point_rel_new)       
    rnewlist.append(TPGMMGMR.reproduce(pmat_new, starting_point_rel_new))

print("Lenght of the new list: ",len(rnewlist))
print("Lenght of A new: ",len(demos_A_new))
#plot the new trajectories
plt.figure()
for i in range(len(demos_A_new)):
    # plt.figure()
    plt.plot(print_points_frames_new[i][0:2,0],print_points_frames_new[i][0:2,1], linewidth=10, alpha=0.9, c='green')
    plt.plot(print_points_frames_new[i][2:4,0],print_points_frames_new[i][2:4,1], linewidth=10, alpha=0.9, c=[30.0/256.0,144.0/256.0,255.0/256.0])
    plt.scatter(print_points_frames_new[i][0,0],print_points_frames_new[i][0,1], linewidth=10, alpha=0.9, c='green')
    plt.scatter(print_points_frames_new[i][2,0],print_points_frames_new[i][2,1], linewidth=10, alpha=0.9, c= [30.0/256.0,144.0/256.0,255.0/256.0])
    plt.plot(rnewlist[i].Data[1,:], rnewlist[i].Data[2,:], 'b')
plt.show()