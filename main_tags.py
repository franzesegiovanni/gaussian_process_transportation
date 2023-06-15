"""
Authors: Ravi Prakash & Giovanni Franzese, March 2023
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#%%
import warnings
warnings.filterwarnings("ignore")
from GILoSA.modules import GILoSA
#from GILoSA.modules import GILoSA_surface
import time
from geometry_msgs.msg import PoseStamped
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

#%%
if __name__ == '__main__':
    GILoSA=GILoSA()
    GILoSA.connect_ROS()
    time.sleep(1)
    # GILoSA.home_gripper()

#%%
time.sleep(1)
print("Record of the source disributions")
GILoSA.go_to_pose(GILoSA.view_marker)
time.sleep(2)
#%%
GILoSA.record_source_distribution()    
#%%
time.sleep(1)
print("Record of the cartesian trajectory")
GILoSA.Record_Demonstration()   
#%%

time.sleep(1)
print("Record of the target disributions")
GILoSA.go_to_pose(GILoSA.view_marker)
#%%
time.sleep(2)
GILoSA.record_target_distribution()    

#%%
time.sleep(1)
print("Save the data") 
GILoSA.save()
GILoSA.save_distributions()

#%%
time.sleep(1)
print("Load the data") 
GILoSA.load()   
GILoSA.load_distributions()

#%% 
time.sleep(1)
print("Train the Gaussian Process Models")
GILoSA.Train_GPs()
#%%
time.sleep(1)
print("Reset to the starting cartesian position if you loaded the demo")
start = PoseStamped()
# GILoSA.home_gripper()

start.pose.position.x = GILoSA.training_traj[0,0]
start.pose.position.y = GILoSA.training_traj[0,1]
start.pose.position.z = GILoSA.training_traj[0,2]

start.pose.orientation.w = GILoSA.training_ori[0,0] 
start.pose.orientation.x = GILoSA.training_ori[0,1] 
start.pose.orientation.y = GILoSA.training_ori[0,2] 
start.pose.orientation.z = GILoSA.training_ori[0,3] 
GILoSA.go_to_pose(start)


#%% 
time.sleep(1)
print("Interactive Control through source distribution")
GILoSA.Interactive_Control(verboose=False)


#%%
time.sleep(1)
print("Find the transported policy")
GILoSA.kernel_transport=C(0.1,[0.1,0.1]) * RBF(length_scale=[0.3], length_scale_bounds=[0.2,0.5]) + WhiteKernel(0.0001, [0.0001,0.0001])
GILoSA.fit_trasportation()
GILoSA.apply_trasportation()
GILoSA.Train_GPs() # Train your policy after transporting the trajectory and the deltas
#%%

time.sleep(1)
print("Reset to the starting cartesian position if you loaded the demo")
start = PoseStamped()
# GILoSA.home_gripper()

start.pose.position.x = GILoSA.training_traj[0,0]
start.pose.position.y = GILoSA.training_traj[0,1]
start.pose.position.z = GILoSA.training_traj[0,2]

start.pose.orientation.w = GILoSA.training_ori[0,0] 
start.pose.orientation.x = GILoSA.training_ori[0,1] 
start.pose.orientation.y = GILoSA.training_ori[0,2] 
start.pose.orientation.z = GILoSA.training_ori[0,3] 
GILoSA.go_to_pose(start)

#%% 
time.sleep(1)
print("Interactive Control through target distribution")
GILoSA.Interactive_Control(verboose=True)
#%%