"""
Authors: Giovanni Franzese and Ravi Prakash
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
This is the code used for the experiment of reshalving 
"""
#%%
from modules import GPT_tag
import time
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import numpy as np
import pickle
#%%
if __name__ == '__main__':
    gpt=GPT_tag()
    gpt.connect_ROS()
    time.sleep(1)
    gpt.home_gripper()
    # Record the trajectory to scan the environment
    #%%
    gpt.record_traj_tags()
    gpt.save_traj_tags()
    #%%
    gpt.load_traj_tags()
    #%% Learn the current tag distribution by moving around the environment
    gpt.source_distribution=[]
    gpt.target_distribution=[]
    gpt.source_distribution=gpt.record_tags(gpt.source_distribution)
    #%%
    print("Source len",len(gpt.source_distribution))
    print("Save the  source distribution data") 
    gpt.save_distributions()  # we are saving both the distributions but only the source is not empty
    # Save source configuration for data analysis later on
    f = open("data/source.pkl","wb")  
    pickle.dump(gpt.source_distribution,f)  
    f.close()   
    #%% Provide the kinesthetic demonstration of the task
    time.sleep(1)
    print("Record of the cartesian trajectory")
    gpt.Record_Demonstration()  
    gpt.save(file="plant_1")
    #%%
    time.sleep(1)
    print("Record of the cartesian trajectory")
    gpt.Record_Demonstration()  
    gpt.save(file="plant_2")
    #%%
    time.sleep(1)
    print("Record of the cartesian trajectory")
    gpt.Record_Demonstration()  
    gpt.save(file="plant_3")
    #%% Start of the experiments
    #%%
    gpt.load_distributions() #you need to re-load the distribution because that is in a particular format and then it is coverget after and overwritten inside the class
    gpt.load(file="plant_1")
    gpt.load_traj_tags()
    gpt.home_gripper()
    #%%
    gpt.load_distributions() #you need to re-load the distribution because that is in a particular format and then it is coverget after and overwritten inside the class
    gpt.load(file="plant_2")
    gpt.load_traj_tags()
    gpt.home_gripper()
    #%%
    gpt.load_distributions() #you need to re-load the distribution because that is in a particular format and then it is coverget after and overwritten inside the class
    gpt.load(file="plant_3")
    gpt.load_traj_tags()
    gpt.home_gripper()
    #%%
    gpt.target_distribution=[]
    gpt.target_distribution=gpt.record_tags(gpt.target_distribution)
    print("Target len", len(gpt.target_distribution) )
    #%%
    if type(gpt.target_distribution) != type(gpt.source_distribution):
        raise TypeError("Both the distribution must be a numpy array.")
    elif not(isinstance(gpt.target_distribution, np.ndarray)) and not(isinstance(gpt.source_distribution, np.ndarray)):
        gpt.convert_distribution_to_array(use_orientation=False)
    #%%
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # # Plot Source distribution
    # ax.scatter(gpt.source_distribution[:,0], gpt.source_distribution[:,1], gpt.source_distribution[:,2], label='Source')
    # for i in range(gpt.source_distribution.shape[0]):
    #     ax.text(gpt.source_distribution[i, 0], gpt.source_distribution[i, 1], gpt.source_distribution[i, 2], '%d' % i)

    # # Plot Target distribution
    # ax.scatter(gpt.target_distribution[:,0], gpt.target_distribution[:,1], gpt.target_distribution[:,2], label='Target')
    # for i in range(gpt.target_distribution.shape[0]):
    #     ax.text(gpt.target_distribution[i, 0], gpt.target_distribution[i, 1], gpt.target_distribution[i, 2], '%d' % i)

    # Plot Training trajectory
    # ax.scatter(gpt.training_traj[:,0], gpt.training_traj[:,1], gpt.training_traj[:,2], label='Demonstration')
    #%%
    time.sleep(1)
    print("Find the transported policy")
    gpt.kernel_transport=C(0.1) * RBF(length_scale=[0.1],  length_scale_bounds=[0.1,1]) + WhiteKernel(0.0000001, noise_level_bounds=[1e-10,1e-5])
    # gpt.kernel_transport=C(0.1) * RBF(length_scale=[0.1]) + WhiteKernel(0.0000001)
    gpt.fit_transportation(do_rotation=True)
    gpt.apply_transportation()
    #%%
    gpt.go_to_start()
    #%%
    print("Interactive Control Starting")
    gpt.control()
# %%
