import os 
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
from models.model_tp_gmm import Multiple_reference_frames_TPGMM
from generate_random_frame_orientation import generate_frame_orientation  
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2) 

# Experiments
number_repetitions=20
minimum_demonstrations=5 #numbber of demonstrations to use to train the gmm model
maximum_demonstrations=8 #numbber of maximum demonstration that can be be used in the model plus one

# Create an empty one-dimensional list with demonstrations using list comprehension
results_df = [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_area= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_dtw= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_fde= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_fad= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
name = ['TP-GMM_{}'.format(i) for i in range(minimum_demonstrations, maximum_demonstrations)]

script_path = str(os.path.dirname(__file__))
filename = script_path + '/data/' + 'reach_target'
policy=Multiple_reference_frames_TPGMM()
policy.load_data(filename)


all_indexes = list(range(len(policy.demos_x)))

for i in range(maximum_demonstrations-minimum_demonstrations):
    for j in range(number_repetitions):
        
        # Use random.sample to select 'm' indexes randomly
        selected_indexes = random.sample(all_indexes, i+minimum_demonstrations)

        # Create a list of non-selected indexes
        non_selected_indexes = [index for index in all_indexes if index not in selected_indexes]
        policy.train(index_partial_dataset=selected_indexes)
        fig, ax = plt.subplots()
        ax=None
        df, area, dtw, fde, fad= policy.reproduce(index_partial_dataset=non_selected_indexes, ax=ax, compute_metrics=True)      
        results_df[i]=results_df[i]+df
        results_area[i]=results_area[i]+area
        results_dtw[i]=results_dtw[i]+dtw
        results_fde[i]=results_fde[i]+fde
        results_fad[i]= results_fad[i]+fad
        # plt.show()
# # print the length of the lists
# print("Frechet")
# print(len(results_df[0]), len(results_df[1]), len(results_df[2]), len(results_df[3]), len(results_df[4]))
# print("Area")
# print(len(results_area[0]), len(results_area[1]), len(results_area[2]), len(results_area[3]), len(results_area[4]))
# print("DTW")
# print(len(results_dtw[0]), len(results_dtw[1]), len(results_dtw[2]), len(results_dtw[3]), len(results_dtw[4]))
# print("FDE")
# print(len(results_fde[0]), len(results_fde[1]), len(results_fde[2]), len(results_fde[3]), len(results_fde[4]))
# print("FDA")
# print(len(results_fad[0]), len(results_fad[1]), len(results_fad[2]), len(results_fad[3]), len(results_fad[4]))

np.savez(script_path + '/results/tpgmm_dataset.npz', 
    results_df=results_df, 
    results_area=results_area, 
    results_dtw=results_dtw,
    results_fde=results_fde, 
    results_fad=results_fad, 
    name=name)

# Out of distribution tests


demos = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]


demos_A_new= demos['A']
demos_b_new = demos['b']

results_fde_new= [[]]
results_fad_new= [[]]

# fig, ax = plt.subplots()
ax=None
for j in range(number_repetitions):
    for i in range(len(demos_A_new)):
        # A, b = demos_A_xdx[i][0], demos_b_xdx[i][0]
        demos_A_new, demos_b_new = generate_frame_orientation(filename)
        A, b =demos_A_new[i][0], demos_b_new[i][0]
        start=policy.starting_point_rel[i] + demos_b_new[i][0][0]
        fde, fad = policy.generalize(A, b, start, ax=ax, final_distance_label=policy.final_distance[i], final_angle_label=policy.final_orientation[i])
        results_fde_new[0].append(fde)
        results_fad_new[0].append(fad)

name = ['TP-GMM9']
np.savez(script_path+ '/results/tpgmm_out_distribution.npz', 
    results_fde=results_fde_new,
    results_fad=results_fad_new, 
    name=name)
# plt.show()