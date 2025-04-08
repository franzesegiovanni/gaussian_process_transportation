"""
Authors: Giovanni Franzese and Ravi Prakash
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
This is the code used for the experiment of reshalving 
"""
#%%
from modules import GPT_DINO
import time
import rospy
from sklearn.gaussian_process.kernels import RBF,WhiteKernel, ConstantKernel as C
import numpy as np
import pickle
import matplotlib.pyplot as plt
#%%
if __name__ == '__main__':
    kernel_transport=C(0.1) * RBF(length_scale=[0.1, 0.2, 0.3],  
                                  length_scale_bounds=[(0.02, 0.1), (0.02, 0.1), (0.2, 1)]) + WhiteKernel(0.0000001, [0.000001,0.00001])
    gpt=GPT_DINO()
    gpt.set_kernel(kernel_transport)
    gpt.connect_ROS()
    time.sleep(1)
    gpt.home()
    rospy.sleep(1)
    gpt.offset_compensator(steps=20)
    gpt.home_gripper()
    # Record the trajectory to scan the environment
    #%% Provide the kinesthetic demonstration of the task
    time.sleep(1)
    print("Record of the cartesian trajectory")
    gpt.Record_Demonstration()  
    gpt.save(file="lattuce_on_cheese")
    #%% Send the robot home
    gpt.home()
    gpt.offset_compensator(steps=20)
    #%% Start of the experiments
    i=0
    #%%
    gpt.load_distributions() #you need to re-load the distribution because that is in a particular format and then it is coverget after and overwritten inside the class
    gpt.load(file="lattuce_on_cheese")
    #%%
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gpt.source_distribution[:,0], gpt.source_distribution[:,1], gpt.source_distribution[:,2], label="Source")
    ax.scatter(gpt.target_distribution[:,0], gpt.target_distribution[:,1], gpt.target_distribution[:,2], label="Target")
    ax.scatter(gpt.training_traj[:,0], gpt.training_traj[:,1], gpt.training_traj[:,2], label="Demonstration", color="green")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    #%%
    time.sleep(1)
    print("Find the transported policy")
    gpt.fit_transportation(do_scale=False, do_rotation=False)
    gpt.apply_transportation()
    #%%
    gpt.go_to_start()
    #%%
    print("Interactive Control Starting")
    gpt.control()    
    #%% Send the robot home
    gpt.home()
    gpt.offset_compensator(steps=20)
