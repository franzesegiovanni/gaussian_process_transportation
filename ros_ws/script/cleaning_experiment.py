"""
Authors: Ravi Prakash & Giovanni Franzese, March 2023
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#%%
import warnings
import torch 
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")
from modules import GPT_surface
import time
from geometry_msgs.msg import PoseStamped
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import rospy
from std_msgs.msg import Bool
import pickle
start_recording=rospy.Publisher('/recording', Bool, queue_size=0)
#%%
if __name__ == '__main__':
    gpt=GPT_surface()
    gpt.connect_ROS()
    time.sleep(5)
    #gpt.home_gripper()
#%%
time.sleep(1)
print("Record of the source disributions")
gpt.go_to_pose(gpt.view_marker)
# time.sleep(2)
#%%
# %matplotlib qt
gpt.record_source_distribution()   
f = open("results/cleaning/source.pkl","wb")
pickle.dump(gpt.source_distribution,f)
# close file
f.close()  
#%%
time.sleep(1)
print("Record of the cartesian trajectory")
gpt.Record_Demonstration()  
gpt.training_traj[:,2]=gpt.training_traj[:,2]-0.025 
# Save traj configuration
f = open("results/cleaning/traj_demo.pkl","wb")
# write the python object (dict) to pickle file
pickle.dump(gpt.training_traj,f)
# close file
f.close()

f = open("results/cleaning/traj_demo_ori.pkl","wb")
# write the python object (dict) to pickle file
pickle.dump(gpt.training_ori,f)
# close file
f.close()
#%%

#%%
time.sleep(1)
print("Reset to the starting cartesian position if you loaded the demo")
start = PoseStamped()

start.pose.position.x = gpt.training_traj[0,0]
start.pose.position.y = gpt.training_traj[0,1]
start.pose.position.z = gpt.training_traj[0,2]

start.pose.orientation.w = gpt.training_ori[0,0] 
start.pose.orientation.x = gpt.training_ori[0,1] 
start.pose.orientation.y = gpt.training_ori[0,2] 
start.pose.orientation.z = gpt.training_ori[0,3] 
gpt.go_to_pose(start)

#%% 
start=Bool()
start.data=True
start_recording.publish(start)
gpt.index=0

gpt.mu_index=0
time.sleep(1)
print("Interactive Control through target distribution")
gpt.Interactive_Control()
#%%
time.sleep(1)
print("Record of the target disributions")
gpt.go_to_pose(gpt.view_marker)
#%%
time.sleep(2)
gpt.target_distribution=gpt.source_distribution

#%%
time.sleep(1)
print("Save the data") 
gpt.save()
gpt.save_distributions()

#%%START EXPERIMENT 
i=0
#%%
time.sleep(1)
gpt.go_to_pose(gpt.view_marker)
#%%
time.sleep(1)
print("Load the data") 
gpt.load()   
gpt.load_distributions()

#%%
time.sleep(2)
gpt.record_target_distribution()  
f = open("results/cleaning/target_"+str(i)+".pkl","wb")
pickle.dump(gpt.target_distribution,f)
# close file
f.close() 
#%%
time.sleep(1)
print("Find the transported policy")
# gpt.kernel_transport=C(0.1) * RBF(length_scale=[0.1, 0.1, 0.1]) + WhiteKernel(0.0001)
gpt.fit_transportation(num_epochs=5)
gpt.apply_transportation()
gpt.Train_GPs() # Train your policy after transporting the trajectory and the deltas
#%%
# Save traj configuration
f = open("results/cleaning/traj_"+str(i)+".pkl","wb")
# write the python object (dict) to pickle file
pickle.dump(gpt.training_traj,f)
# close file
f.close()

f = open("results/cleaning/traj_ori"+str(i)+".pkl","wb")
# write the python object (dict) to pickle file
pickle.dump(gpt.training_ori,f)
# close file
f.close()
#%%
time.sleep(1)
print("Reset to the starting cartesian position if you loaded the demo")
start = PoseStamped()

start.pose.position.x = gpt.training_traj[0,0]
start.pose.position.y = gpt.training_traj[0,1]
start.pose.position.z = gpt.training_traj[0,2]

start.pose.orientation.w = gpt.training_ori[0,0] 
start.pose.orientation.x = gpt.training_ori[0,1] 
start.pose.orientation.y = gpt.training_ori[0,2] 
start.pose.orientation.z = gpt.training_ori[0,3] 
gpt.go_to_pose(start)

#%% 
start=Bool()
start.data=True
start_recording.publish(start)
gpt.index=0

gpt.mu_index=0
time.sleep(1)
print("Interactive Control through target distribution")
gpt.Interactive_Control()
i=i+1
# %%