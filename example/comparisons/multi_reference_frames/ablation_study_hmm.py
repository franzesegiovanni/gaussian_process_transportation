import numpy as np
import matplotlib.pyplot as plt
import random
from generate_random_frame_orientation import generate_frame_orientation  
from models.model_hmm import Multiple_reference_frames_HMM
import os
# Experiments
number_repetitions=20
minimum_demonstrations=5 #numbber of demonstrations to use to train the gmm model
maximum_demonstrations=8 # maximum number that can be used to trained the model plus one
# Create an empty one-dimensional list with demonstrations using list comprehension
results_df = [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_area= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_dtw= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_fde= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_fad= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
name = ['HMM_{}'.format(i) for i in range(minimum_demonstrations, maximum_demonstrations)]
script_path = str(os.path.dirname(__file__))
filename = script_path + '/data/' + 'reach_target'
policy=Multiple_reference_frames_HMM()
policy.load_data(filename)

demos_xdx_augm=policy.demos_xdx_augm
demos_xdx=policy.demos_xdx
demos_A_xdx=policy.demos_A_xdx
demos_b_xdx=policy.demos_b_xdx
demos_x=policy.demos_x
demos_A=policy.demos_A
demos_b=policy.demos_b
final_distance=np.zeros((len(demos_x),2))
final_orientation=np.zeros((len(demos_x),1))


for i in range(len(demos_x)):
    final_distance[i]=  np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,:] - demos_b[i][0][1])
    final_delta=np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,:]-demos_x[i][-2,:])
    final_orientation[i]= np.arctan2(final_delta[1],final_delta[0])

for i in range(maximum_demonstrations-minimum_demonstrations):
    for j in range(number_repetitions):
        sampled_demo=random.sample(demos_xdx_augm, i+minimum_demonstrations)
        indices = [demos_xdx_augm.index(element) for element in sampled_demo]
        not_in_indices = [index for index in range(len(demos_xdx_augm)) if index not in indices]
        policy.demos_xdx_augm=sampled_demo
        policy.train()

        # fig, ax = plt.subplots()
        for k in not_in_indices:    
            A, b = demos_A_xdx[k][0], demos_b_xdx[k][0]
            start=demos_xdx[k][0]
            df, area, dtw, fde, fad= policy.reproduce(k, compute_metrics=True)

            results_df[i].append(df)
            results_area[i].append(area)
            results_dtw[i].append(dtw)
            results_fde[i].append(fde)
            results_fad[i].append(fad)

# plt.show()

np.savez(script_path + '/results/hmm_dataset.npz', 
    results_df=results_df, 
    results_area=results_area, 
    results_dtw=results_dtw,
    results_fde=results_fde, 
    results_fad=results_fad, 
    name=name)

# Create random orientation of the frames

results_df=np.zeros( (  number_repetitions, len(demos_x)) )
results_area=np.zeros(( number_repetitions, len(demos_x)) )
results_dtw=np.zeros((  number_repetitions, len(demos_x) ))
results_fde=np.zeros((  number_repetitions , len(demos_x) ))

results_fde_new= [[]]
results_fad_new= [[]]
#we use always all the demos in the training set and we compute the error to reach the final point in a new situation 
filename = script_path + '/data/' + 'reach_target'
policy=Multiple_reference_frames_HMM()
policy.load_data(filename)
policy.train()
policy.find_the_most_probable_sequence()
for j in range(number_repetitions):
    # fig, ax = plt.subplots()
    ax=None

    demos_A_new, demos_b_new = generate_frame_orientation(filename)
    demos_A_xdx_new = [np.kron(np.eye(2), d) for d in demos_A_new]
    demos_b_xdx_new = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b_new]

    for k in range(9):
        A, b = demos_A_xdx_new[k][0], demos_b_xdx_new[k][0]
        start=np.zeros_like(demos_xdx[k][0])
        start=demos_xdx[k][0]+(demos_b_xdx_new[k][0][0]-demos_b_xdx[k][0][0])
        vel_new=A[0][2:4,2:4] @ np.linalg.inv(demos_A_xdx[k][0][0][2:4,2:4]) @ start[2:]
        start[2:]=vel_new

        fde, fad = policy.generalize(A, b, start, ax=ax, final_distance_label=final_distance[k], final_distance_angle=final_orientation[k])
        results_fde_new[0].append(fde)
        results_fad_new[0].append(fad)

name = ['HMM_9']
np.savez(script_path +  '/results/hmm_out_distribution.npz', 
    results_fde=results_fde_new,
    results_fad=results_fad_new,
    name=name)

# plt.show()