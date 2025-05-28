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
from geometry_msgs.msg import PoseStamped
import rospy
from sklearn.gaussian_process.kernels import RBF, Matern,WhiteKernel, ConstantKernel as C
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
    gpt.save()
    #%% Save the teaching trajectory
    f = open("data/traj_demo.pkl","wb")
    pickle.dump(gpt.training_traj,f)
    f.close()  
    f = open("data/traj_demo_ori.pkl","wb")
    pickle.dump(gpt.training_ori,f)
    f.close()
    #%% Start of the experiments
    i=0
    #%%
    gpt.load_distributions() #you need to re-load the distribution because that is in a particular format and then it is coverget after and overwritten inside the class
    gpt.load()
    gpt.load_traj_tags()
    gpt.home_gripper()
    #%%
    gpt.target_distribution=[]
    gpt.target_distribution=gpt.record_tags(gpt.target_distribution)
    print("Target len", len(gpt.target_distribution) )
    #%%
    # Save target distribution for data analysis later on
    f = open("data/target_"+str(i)+".pkl","wb")   
    pickle.dump(gpt.target_distribution,f)
    f.close()   
    #%%
    if type(gpt.target_distribution) != type(gpt.source_distribution):
        raise TypeError("Both the distribution must be a numpy array.")
    elif not(isinstance(gpt.target_distribution, np.ndarray)) and not(isinstance(gpt.source_distribution, np.ndarray)):
        gpt.convert_distribution_to_array(use_orientation=True)

    #%%
    time.sleep(1)
    print("Find the transported policy")
    gpt.kernel_transport=C(0.1) * RBF(length_scale=[0.1],  length_scale_bounds=[0.1,0.5]) + WhiteKernel(0.0000001, [0.0000001,0.0000001])
    gpt.fit_transportation()
    gpt.apply_transportation()
    #%%
    # Save transported trajectory and orientation for data analysis
    f = open("data/traj_"+str(i)+".pkl","wb") 
    pickle.dump(gpt.training_traj,f)
    f.close()
    f = open("data/traj_ori_"+str(i)+".pkl","wb")
    pickle.dump(gpt.training_ori,f) 
    f.close()
    #%%
    gpt.go_to_start()
    #%%
    print("Interactive Control Starting")
    gpt.control()
    i=i+1    
    #%% Make the robot passive and reset the environment 
    gpt.Passive()
# %%
