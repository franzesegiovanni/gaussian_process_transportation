import os 
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
from models.model_laplacian_editing import Multiple_Reference_Frames_LA
from generate_random_frame_orientation import generate_frame_orientation  
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2) 

# Experiments
number_repetitions=20
minimum_demonstrations=2 #numbber of demonstrations to use to train the gmm model
maximum_demonstrations=9 #numbber of demonstrations that are available in total

# Create an empty one-dimensional list with demonstrations using list comprehension
results_df = [[]]
results_area=[[]]
results_dtw= [[]]
results_fde= [[]]
results_fad= [[]]
name= ['LE']
script_path = str(os.path.dirname(__file__))
filename = script_path + '/data/' + 'reach_target'
policy=Multiple_Reference_Frames_LA()
policy.load_dataset(filename, use_extra_points=False)

for j in range(number_repetitions):
    # fig, ax = plt.subplots()
    ax=None
    index_source = random.choice(range(len(policy.demos_x))) 
    vector = [i for i in range(len(policy.demos_x)) if i != index_source] 
    for k in vector:
        df, area, dtw, fde, fda= policy.reproduce(index_source, index_target=k, compute_metrics=True, ax=ax)
        results_df[0].append(df)
        results_area[0].append(area)
        results_dtw[0].append(dtw)
        results_fde[0].append(fde)
        results_fad[0].append(fda)

# plt.show()

np.savez(script_path + '/results/la_dataset.npz', 
    results_df=results_df, 
    results_area=results_area, 
    results_dtw=results_dtw,
    results_fde=results_fde, 
    results_fad=results_fad, 
    name=name)

results_fda_new=[[]]
results_fde_new=[[]]

for j in range(number_repetitions):
    # fig, ax = plt.subplots()
    ax=None
    A_new, b_new = generate_frame_orientation(filename)
    index_source = random.choice(range(len(policy.demos_A)))  
    policy.load_test_dataset(A_new, b_new, use_extra_points=False)
    for k in range(len(A_new)):
        fde, fda = policy.generalize(index_source, k,  ax=ax, compute_metrics=True )
        results_fde_new[0].append(fde)
        results_fda_new[0].append(fda)

np.savez(script_path + '/results/la_out_distribution.npz', 
    results_fde=results_fde_new,
    results_fad=results_fda_new, 
    name=name)
# plt.show()