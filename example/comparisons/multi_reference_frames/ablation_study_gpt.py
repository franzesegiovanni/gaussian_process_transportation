import os 
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
from models.model_gpt import Multiple_Reference_Frames_GPT
from generate_random_frame_orientation import generate_frame_orientation  
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2) 

# Experiments
number_repetitions=20
minimum_demonstrations=2 #numbber of demonstrations to use to train the gmm model
maximum_demonstrations=9 #numbber of demonstrations that are available in total

# Create an empty one-dimensional list with demonstrations using list comprehension
results_df = [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_area= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_dtw= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_fde= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]
results_fda= [[] for _ in range(maximum_demonstrations-minimum_demonstrations)]

script_path = str(os.path.dirname(__file__))
filename = script_path + '/data/' + 'reach_target'
policy=Multiple_Reference_Frames_GPT()
policy.load_dataset(filename)
policy.train()

for j in range(number_repetitions):
    # fig, ax = plt.subplots()
    ax=None
    index_source = random.choice(range(len(policy.demos_x))) 
    vector = [i for i in range(len(policy.demos_x)) if i != index_source] 
    for k in vector:
        df, area, dtw, fde, fda= policy.reproduce(index_source, index_target=k, compute_metrics=True, ax=ax)
        results_df.append(df)
        results_area.append(area)
        results_dtw.append(dtw)
        results_fde.append(fde)
        results_fda.append(fda)

# plt.show()

np.savez(script_path + '/results/gpt_dataset.npz', 
    results_df=results_df, 
    results_area=results_area, 
    results_dtw=results_dtw,
    results_fde=results_fde, 
    results_fda=results_fda)

results_fda_new=[]
results_fde_new=[]

for j in range(number_repetitions):
    # fig, ax = plt.subplots()
    ax=None
    A_new, b_new = generate_frame_orientation(filename)
    index_source = random.choice(range(len(policy.demos_A)))  
    policy.load_test_dataset(A_new, b_new)
    for k in range(len(A_new)):
        fde, fda = policy.generalize(index_source, k,  ax=ax, compute_metrics=True )
        results_fde_new.append(fde)
        results_fda_new.append(fda)

np.savez(script_path + '/results/gpt_out_distribution.npz', 
    results_fde=results_fde_new,
    results_fad=results_fda_new)
# plt.show()