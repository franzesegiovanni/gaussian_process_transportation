import numpy as np
import os
import copy
def random_rotation_matrix_2d(magnitude=0.1):
    theta = np.random.uniform(-magnitude*np.pi, magnitude*np.pi)  # Random angle between 0 and 2*pi

    c = np.cos(theta)
    s = np.sin(theta)

    rotation_matrix = np.array([[c, -s],
                                [s, c]])

    return rotation_matrix

def generate_frame_orientation(filename):

    demos = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]

    ### Trajectory data
    demos_x = demos['x'] # position

    ### Coordinate systems transformation
    demos_A = [d for d in demos['A']]
    demos_b = [d for d in demos['b']]

    demos_A_new=copy.deepcopy(demos_A)
    demos_b_new=copy.deepcopy(demos_b)
    translation_offeset=20
    for i in range (len(demos_x)):
        for j in range(2):
                bTransform = (translation_offeset*np.random.randn(2, 1) - translation_offeset/2).reshape(-1,)
                aTransform = random_rotation_matrix_2d(magnitude=0.5) 
                demos_A_new[i][0][j]= aTransform @  demos_A[i][0][j]
                demos_b_new[i][0][j] = demos_b_new[i][0][j]+bTransform

    return demos_A_new, demos_b_new

if __name__ == '__main__':
    demos_A_new, demos_b_new = generate_frame_orientation()
    np.save('demos_A.npy', np.array(demos_A_new, dtype=object) , allow_pickle=True)
    np.save('demos_b.npy', np.array(demos_b_new, dtype=object), allow_pickle=True)