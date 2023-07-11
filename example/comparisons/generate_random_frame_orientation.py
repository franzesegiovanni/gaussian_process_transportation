import numpy as np
import os

def random_rotation_matrix_2d(magnitude=0.1):
    theta = np.random.uniform(-magnitude*np.pi, magnitude*np.pi)  # Random angle between 0 and 2*pi

    c = np.cos(theta)
    s = np.sin(theta)

    rotation_matrix = np.array([[c, -s],
                                [s, c]])

    return rotation_matrix

filename = 'reach_target'

pbd_path = os. getcwd()  + '/data_hmm/'

demos = np.load(pbd_path + filename + '.npy', allow_pickle=True, encoding='latin1')[()]

### Trajectory data
demos_x = demos['x'] # position
demos_dx = demos['dx'] # velocity
demos_xdx = [np.concatenate([x, dx], axis=1) for x, dx in zip(demos_x, demos_dx)] # concatenation

### Coordinate systems transformation
demos_A = [d for d in demos['A']]
demos_b = [d for d in demos['b']]


# distribution_new=np.zeros((len(demos_x),len(demos_x[0])*len(demos_x[0]),2))
distribution=np.zeros((len(demos_x),4,2))
distribution_new=np.zeros((len(demos_x),4,2))
index=2
for i in range(len(demos_x)):
    distribution[i,0,:]=demos_b[i][0][0]
    distribution[i,1,:]=demos_b[i][0][0]+demos_A[i][0][0] @ np.array([ 0, 10])
    distribution[i,2,:]=demos_b[i][0][1]
    distribution[i,3,:]=demos_b[i][0][1]+demos_A[i][0][1] @ np.array([ 0, -10])

A=np.empty((len(demos_x), 2, 2, 2))
B=np.empty((len(demos_x), 2, 2))
demos_A_new=np.copy(demos_A)
demos_b_new=np.copy(demos_b)
translation_offeset=30
for i in range (len(demos_x)):
    for j in range(2):
            bTransform = (translation_offeset*np.random.rand(2, 1) - translation_offeset/2).reshape(-1,)
            aTransform = random_rotation_matrix_2d(magnitude=0.5) 
            demos_A_new[i][0][j]= aTransform @  demos_A[i][0][j]
            demos_b_new[i][0][j] = demos_b_new[i][0][j]+bTransform

np.save('demos_A.npy', demos_A_new, allow_pickle=True)
np.save('demos_b.npy', demos_b_new, allow_pickle=True)