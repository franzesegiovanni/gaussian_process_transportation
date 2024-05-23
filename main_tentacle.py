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
from sensors.tag_detector import save_source, save_target, load_multiple_sources, load_target
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
    gpt.source_distribution=gpt.record_tags(gpt.source_distribution)
    #%%
    print("Source len",len(gpt.source_distribution))
    print("Save the  source distribution data") 
    save_source(gpt.source_distribution)
    # Save source configuration for data analysis later on
    #%% Provide the kinesthetic demonstration of the task
    gpt.Record_Demonstration()  
    gpt.save_demo() 
    #%%
    #%%
    gpt.target_distribution=[]
    gpt.target_distribution=gpt.record_tags(gpt.target_distribution)
    print("Target len", len(gpt.target_distribution) )

    save_target(gpt.target_distribution)
    #%%
    source_distribution, target_distribution, index= find_closest_source_to_target(use_orientation=False)

    #%% 
    gpt.source_distribution = source_distribution
    gpt.target_distribution = target_distribution
    gpt.load_demo(index=index)
    #%%
    time.sleep(1)
    print("Find the transported policy")
    gpt.kernel_transport=C(0.1) * RBF(length_scale=[0.1],  length_scale_bounds=[0.1,0.5]) + WhiteKernel(0.0000001, [0.0000001,0.0000001])
    gpt.fit_transportation()
    gpt.apply_transportation()
    #%%
    gpt.go_to_start()
    #%%
    print("Interactive Control Starting")
    gpt.control()
    i=i+1    
    #%% Make the robot passive and reset the environment 
    gpt.Passive()
# %%
