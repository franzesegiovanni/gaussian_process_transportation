import os
import pickle 
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#load source distribution
file_dir= os.path.dirname(__file__)
from convert_tag_2_dist import convert_distribution

data_folder =file_dir+ "/results/dressing/"
target = []
traj = []
traj_ori = []
name_target = []
name_traj = []
name_traj_ori = []
def load(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data  

traj_demo= load(file_dir+ "/results/dressing/traj_demo.pkl")
traj_demo_ori= load(file_dir+ "/results/dressing/traj_demo_ori.pkl")

source_tags = load(data_folder + "/source.pkl")


for filename in sorted(os.listdir(data_folder)):
    if filename.endswith(".pkl") and not "demo" in filename:
        if "target" in filename:
            name_target.append(os.path.splitext(filename)[0])
            target_tags= load(data_folder+filename)
            source_dist, target_dist= convert_distribution(source_tags, target_tags, use_orientation=False)
            target.append(target_dist)
        elif "traj_ori" in filename:
            traj_ori.append(load(data_folder+ filename))
            name_traj_ori.append(os.path.splitext(filename)[0])
        elif "traj" in filename:
            traj.append(load(data_folder+filename))
            name_traj.append(os.path.splitext(filename)[0])


print(name_target)
print(name_traj)
print(name_traj_ori)

plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.plot(traj_demo[:,0], traj_demo[:,1], traj_demo[:,2], '--')
ax.scatter(traj_demo[0,0], traj_demo[0,1], traj_demo[0,2], c='r', marker='o')
ax.scatter(traj_demo[-1,0], traj_demo[-1,1], traj_demo[-1,2], c='g', marker='o')
ax.scatter(source_dist[:,0], source_dist[:,1], source_dist[:,2], c='b', marker='o')


for i, trajectory in enumerate(traj):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2])
    ax.scatter(trajectory[0,0], trajectory[0,1], trajectory[0,2], c='r', marker='o', label='start')
    ax.scatter(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2], c='g', marker='o', label='end')
    ax.scatter(target[i][:,0], target[i][:,1], target[i][:,2], c='b', marker='o')

plt.legend()
plt.show()


