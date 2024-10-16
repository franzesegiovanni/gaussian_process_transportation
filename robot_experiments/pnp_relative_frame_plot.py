#source_april_tags_ravi
import os
import pickle 
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from convert_tag_2_dist import match_markers
from panda_ros.pose_transform_functions import array_quat_2_pose, list_2_quaternion
from utils import relative_transformation
import numpy as np
# compute relative 

file_dir= os.path.dirname(__file__)
data_folder =file_dir+ "/results/pnp/"

fig_folder =file_dir+ "/figures/"



target = []
traj = []
traj_ori = []
name_target = []
name_traj = []
name_traj_ori = []
def load(filename):
    with open(filename,"rb") as file:
        data = pickle.load(file)
    return data  

traj_demo= load(file_dir+ "/results/pnp/traj_demo.pkl")
traj_demo_ori= load(file_dir+ "/results/pnp/traj_demo_ori.pkl")


source_tags = load(file_dir+ "/results/pnp/source.pkl")

for filename in sorted(os.listdir(data_folder)):
    if filename.endswith(".pkl") and not "demo" in filename:
        if "target" in filename:
            name_target.append(os.path.splitext(filename)[0])
            target_tags= load(data_folder+filename)
            source, target_dist= match_markers(source_tags, target_tags)
            target.append(target_dist)
        elif "traj_ori" in filename:
            traj_ori.append(load(data_folder+ filename))
            name_traj_ori.append(os.path.splitext(filename)[0])
        elif "traj" in filename:
            traj.append(load(data_folder+filename))
            name_traj.append(os.path.splitext(filename)[0])

# compute the range of variability in x,y,z of target and object frame
pos_x=[]
pos_y=[]
pos_z=[]
yaw=[]
pos_x2=[]
pos_y2=[]
pos_z2=[]
yaw2=[]

for i in range(len(target)):
    if len(target[i])>1:
        # print("target", target[i][0].pose.position)
        pos_x.append(target[i][0].pose.position.x)
        pos_y.append(target[i][0].pose.position.y)
        pos_z.append(target[i][0].pose.position.z)
        q=target[i][0].pose.orientation
        yaw.append(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1-2*( q.y * q.y + q.z * q.z)))

        pos_x2.append(target[i][1].pose.position.x)
        pos_y2.append(target[i][1].pose.position.y)
        pos_z2.append(target[i][1].pose.position.z)
        q=target[i][1].pose.orientation

pos_x=np.array(pos_x)
pos_y=np.array(pos_y)
pos_z=np.array(pos_z)
yaw=np.array(yaw)

pos_x2=np.array(pos_x2)
pos_y2=np.array(pos_y2)
pos_z2=np.array(pos_z2)

#Compute range of variability of eacho of the previous variables
print("Range of variability of the object frame")
print("pos_x:", np.ptp(pos_x), "meters")
print("pos_y:", np.ptp(pos_y), "meters")
print("pos_z:", np.ptp(pos_z), "meters")
print("yaw:", np.ptp(yaw)*180/np.pi, "degrees")
print("Range of variability of the goal frame")
print("pos_x2:", np.ptp(pos_x2), "meters")
print("pos_y2:", np.ptp(pos_y2), "meters")
print("pos_z2:", np.ptp(pos_z2), "meters")

pose_trajectories=[]
pose_trajectories_relative_0=[]
pose_trajectories_relative_1=[]

for i in range(len(traj)):
    traj_pose=[]
    traj_relative_pose_0=[]
    traj_relative_pose_1=[]
    if len(target[i])>1:
        for j in range(len(traj[i])):
            traj_quat=list_2_quaternion(traj_ori[i][j])
            traj_pose.append(array_quat_2_pose(traj[i][j],traj_quat))

            traj_relative_pose_0.append(relative_transformation(traj_pose[j].pose, target[i][0].pose)) 
            traj_relative_pose_1.append(relative_transformation(traj_pose[j].pose, target[i][1].pose))
        pose_trajectories.append(traj_pose)
        pose_trajectories_relative_0.append(traj_relative_pose_0)
        pose_trajectories_relative_1.append(traj_relative_pose_1)

pose_trajectories_demo=[]

traj_pose=[]
pose_trajectories_relative_0_demo=[]
pose_trajectories_relative_1_demo=[]
for j in range(len(traj_demo)):
    traj_quat=list_2_quaternion(traj_demo_ori[j])
    traj_pose.append(array_quat_2_pose(traj_demo[j],traj_quat))
    pose_trajectories_relative_0_demo.append(relative_transformation(traj_pose[j].pose, source[0].pose)) 
    pose_trajectories_relative_1_demo.append(relative_transformation(traj_pose[j].pose, source[1].pose))

pose_trajectories_demo.append(traj_pose)

relative_0_demo=np.zeros([3,len(pose_trajectories_relative_0_demo)])
relative_1_demo=np.zeros([3,len(pose_trajectories_relative_1_demo)])
for j in range(len(pose_trajectories_relative_0_demo)):
    relative_0_demo[0,j]=pose_trajectories_relative_0_demo[j].position.x
    relative_0_demo[1,j]=pose_trajectories_relative_0_demo[j].position.y
    relative_0_demo[2,j]=pose_trajectories_relative_0_demo[j].position.z

    relative_1_demo[0,j]=pose_trajectories_relative_1_demo[j].position.x
    relative_1_demo[1,j]=pose_trajectories_relative_1_demo[j].position.y
    relative_1_demo[2,j]=pose_trajectories_relative_1_demo[j].position.z

relative_0=np.zeros([len(pose_trajectories_relative_0), 3, len(pose_trajectories_relative_0[0])])
relative_1=np.zeros([len(pose_trajectories_relative_1),3,len(pose_trajectories_relative_1[0])])
for i in range(len(pose_trajectories_relative_0)):
    for j in range(len(pose_trajectories_relative_0[i])):
        relative_0[i,0,j]=pose_trajectories_relative_0[i][j].position.x
        relative_0[i,1,j]=pose_trajectories_relative_0[i][j].position.y
        relative_0[i,2,j]=pose_trajectories_relative_0[i][j].position.z

        relative_1[i,0,j]=pose_trajectories_relative_1[i][j].position.x
        relative_1[i,1,j]=pose_trajectories_relative_1[i][j].position.y
        relative_1[i,2,j]=pose_trajectories_relative_1[i][j].position.z


fig, axs = plt.subplots(3, 2,figsize=(10, 6))
time= np.linspace(0, len(pose_trajectories_relative_0[0])/20, len(pose_trajectories_relative_0[0]))

axs[0, 0].plot(time, relative_0[0,0,:], c='b', label='executions')

for i in range(len(pose_trajectories_relative_0)):
    axs[0, 0].plot(time, relative_0[i,0,:], c='b')
    axs[1, 0].plot(time, relative_0[i,1,:], c='b')
    axs[2, 0].plot(time, relative_0[i,2,:], c='b')

    axs[0, 1].plot(time, relative_1[i,0,:], c='b')
    axs[1, 1].plot(time, relative_1[i,1,:], c='b')
    axs[2, 1].plot(time, relative_1[i,2,:], c='b')

axs[0, 0].plot(time, relative_0_demo[0,:], c='r', linewidth=2, label='demo')
axs[0,0].set_title('Relative to the object', fontsize=20)
axs[0,0].set_ylabel('X [m]', fontsize=20)
axs[1, 0].plot(time, relative_0_demo[1,:], c='r',linewidth=2)
axs[1,0].set_ylabel('Y [m]',fontsize=20)
axs[2, 0].plot(time, relative_0_demo[2,:], c='r',linewidth=2)
axs[2,0].set_ylabel('Z [m]', fontsize=20)
axs[2,0].set_xlabel('Time [s]', fontsize=20)

axs[0, 1].plot(time, relative_1_demo[0,:], c='r',linewidth=2)
axs[0,1].set_title('Relative to the goal', fontsize=20)
axs[1, 1].plot(time, relative_1_demo[1,:], c='r',linewidth=2)
axs[2, 1].plot(time, relative_1_demo[2,:], c='r',linewidth=2)
axs[2,1].set_xlabel('Time [s]', fontsize=20)

axs[0, 0].legend()
ymin = -0.9  # replace with your desired minimum
ymax = 0.9  # replace with your desired maximum

for ax in axs.flat:
    ax.set_ylim([ymin, ymax])
    ax.set_xlim([0, np.max(time)])

axs[0, 0].set_xticks([])
axs[0, 1].set_xticks([])
axs[1, 0].set_xticks([])
axs[1, 1].set_xticks([])
axs[0, 1].set_yticks([])
axs[1, 1].set_yticks([])
axs[2, 1].set_yticks([])

final_time=len(pose_trajectories_relative_0[0])/20
x_value = [0.3* final_time,0.7* final_time]# replace with the x-coordinate of the vertical line
text = ['Pick', 'Place']  # replace with your desired text

for ax in axs.flat:
    for x in x_value:
        ax.axvline(x=x, color='k', linestyle='--')  
axs[0, 0].text(x_value[0]+0.1, ymax, text[0], va='top',rotation=90, fontsize=10)
axs[0, 0].text(x_value[1]+0.1, ymax, text[1], va='top',rotation=90, fontsize=10)

axs[0, 1].text(x_value[0]+0.1, ymax, text[0], va='top',rotation=90, fontsize=10)
axs[0, 1].text(x_value[1]+0.1, ymax, text[1], va='top',rotation=90, fontsize=10)

plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig(fig_folder+'relative_pose.pdf', dpi=300, bbox_inches='tight')
plt.show()




