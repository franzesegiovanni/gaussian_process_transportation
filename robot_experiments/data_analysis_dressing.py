import os
import pickle 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import itertools
#load source distribution
file_dir= os.path.dirname(__file__)
from convert_tag_2_dist import convert_distribution
import numpy as np
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

def sort_points(points):
    permutations = list(itertools.permutations(points))
    cost = []
    for perm in permutations:
        perm= np.array(perm)
        cost.append(np.sum(np.linalg.norm(perm[1:,:]-perm[:-1,:],1)))
    min_index = np.argmin(cost)
    permuted_list = permutations[min_index]
    index = []
    # print(points)
    for i in range(len(points)):
        index.append(int(np.argmin(np.linalg.norm(permuted_list-points[i], axis=1))))
    # index = [int(x) for x in index]  
    print(index)  
    return permuted_list, index


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


plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.plot(traj_demo[:,0], traj_demo[:,1], traj_demo[:,2], '--', label='demo')
ax.scatter(traj_demo[0,0], traj_demo[0,1], traj_demo[0,2], c='r', marker='o', label='start')
ax.scatter(traj_demo[-1,0], traj_demo[-1,1], traj_demo[-1,2], c='g', marker='o', label='end')
source_dist , index = sort_points(source_dist)
source_dist = np.vstack(source_dist)
ax.plot(source_dist[:,0], source_dist[:,1], source_dist[:,2], '--', c='k')
ax.scatter(source_dist[0,0], source_dist[0,1], source_dist[0,2], c='b', marker='o', label='hand')
ax.scatter(source_dist[1,0], source_dist[1,1], source_dist[1,2], c='y', marker='o',label='wrist')
ax.scatter(source_dist[2,0], source_dist[2,1], source_dist[2,2], c='m', marker='o', label='elbow')
ax.scatter(source_dist[3,0], source_dist[3,1], source_dist[3,2], c='c', marker='o', label='shoulder')







for i, trajectory in enumerate(traj):
    # plt.figure()
    # ax = plt.axes(projection='3d')
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2])
    ax.scatter(trajectory[0,0], trajectory[0,1], trajectory[0,2], c='r', marker='o')
    ax.scatter(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2], c='g', marker='o')
    selected_target =np.array( target[i])
    target_sorted = selected_target[index,:]
    ax.plot(target_sorted[:,0], target_sorted[:,1], target_sorted[:,2], c='k', marker='o')
    ax.scatter(target_sorted[0,0], target_sorted[0,1], target_sorted[0,2], c='b', marker='o')
    ax.scatter(target_sorted[1,0], target_sorted[1,1], target_sorted[1,2], c='y', marker='o')
    ax.scatter(target_sorted[2,0], target_sorted[2,1], target_sorted[2,2], c='m', marker='o')
    ax.scatter(target_sorted[3,0], target_sorted[3,1], target_sorted[3,2], c='c', marker='o')

plt.legend()
plt.show()
