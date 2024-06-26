import os
import pickle 
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

file_dir= os.path.dirname(__file__)
data_folder =file_dir+ "/results/cleaning/"


target = []
traj = []
traj_ori = []
execution = []
name_target = []
name_traj = []
name_traj_ori = []
def load(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data  

traj_demo= load(file_dir+ "/results/cleaning/traj_demo.pkl")
traj_demo_ori= load(file_dir+ "/results/cleaning/traj_demo_ori.pkl")

source_dist = load(file_dir+ "/results/cleaning/source.pkl")

for filename in sorted(os.listdir(data_folder)):
    if filename.endswith(".pkl") and not "demo" in filename:
        if "target" in filename:
            name_target.append(os.path.splitext(filename)[0])
            target_dist= load(data_folder+filename)
            target.append(target_dist)
        elif "traj_ori" in filename:
            traj_ori.append(load(data_folder+ filename))
            name_traj_ori.append(os.path.splitext(filename)[0])
        elif "traj" in filename:
            traj.append(load(data_folder+filename))
            name_traj.append(os.path.splitext(filename)[0])
    if filename.endswith(".npz"):
        execution.append(np.load(data_folder+filename))




fig,axs = plt.subplots(1,len(target)+1,subplot_kw={'projection': '3d'},figsize=(20, 5))
ax=axs[0]
ax.scatter(source_dist[:,0], source_dist[:,1], source_dist[:,2], c='b', marker='o', alpha=0.1,label='source')
ax.plot(traj_demo[:,0], traj_demo[:,1], traj_demo[:,2], '--', c='k', linewidth=2, label='demo')
ax.scatter(traj_demo[0,0], traj_demo[0,1], traj_demo[0,2], c='r', marker='o', label='start')
ax.scatter(traj_demo[-1,0], traj_demo[-1,1], traj_demo[-1,2], c='g', marker='o', label='end')
ax.text(x=0.5, y=-0.4, z=-0.2, s='D', fontsize=20, ha='left', va='top')


for i, trajectory in enumerate(traj):
    ax = axs[i%len(target)+1]  # Select the subplot
    # ax = plt.axes(projection='3d')
    ax.scatter(target[i][:,0], target[i][:,1], target[i][:,2], c='b', alpha=0.1)
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], c='k', linewidth=2 )
    ax.scatter(trajectory[0,0], trajectory[0,1], trajectory[0,2], c='r', marker='o')
    ax.scatter(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2], c='g', marker='o')
    ax.text(x=0.5, y=-0.3, z=-0.1, s=str(i+1), fontsize=20, ha='left', va='top')

for i in range(0,len(target)+1):
    
    ax = axs[i]  # Select the subplot
    ax.set_xlim(0.2,0.7)
    ax.set_ylim(-0.3, 0.2 )
    ax.set_zlim( 0,1)

        # Set the point of view
    ax.view_init(elev=30, azim=3)

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    # Make panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Hide the axes and grid
    ax.axis('off')
    ax.grid(False)
plt.subplots_adjust(wspace=-0.1, hspace=-0.6)        
#save figure
plt.savefig(file_dir+ "/figures/cleaning.pdf", dpi=300, bbox_inches='tight') 
plt.figure()

trajectory=execution[0]
time=np.arange(0, len(trajectory['recorded_force_torque'][0,:]))/20
force_norm=np.linalg.norm(trajectory['recorded_force_torque'][0:3,:], axis=0)
plt.plot(time, force_norm, '--',label='D')
i=1
for trajectory in execution[1:]:
    time=np.arange(0, len(trajectory['recorded_force_torque'][0,:]))/20
    force_norm=np.linalg.norm(trajectory['recorded_force_torque'][0:3,:], axis=0)
    plt.plot(time, force_norm, label=str(i))
    i=i+1
plt.legend()
plt.title('Cleaning surface generalization', fontsize=16)
plt.grid(True)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Force [N]', fontsize=14)
# plt.figure()
# for j in range(3):
#     time=np.arange(0, len(execution[i]['recorded_force_torque'][j,:]))/20
#     plt.subplot(3,1,j+1)
#     plt.plot(time, execution[i]['recorded_force_torque'][j,:])
# plt.legend()
plt.savefig(file_dir+ "/figures/force_norm.pdf", dpi=300, bbox_inches='tight') 
# plt.show()

# plot of the forces 


import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA

target.insert(0,source_dist)


hausdorff_dist=np.zeros((len(target), len(target)))
chamfer_dist=np.zeros((len(target), len(target)))
pca_dist=np.zeros((len(target), len(target)))
mse=np.zeros((len(target), len(target)))



for i, source_points in enumerate(target):
    for j, target_points in enumerate(target):

        # 1. Hausdorff Distance
        hausdorff_dist[i,j] = distance.directed_hausdorff(source_points.tolist(), target_points.tolist())[0]

        # 2. Chamfer Distance
        chamfer_dist[i,j]  = np.mean(np.min(distance.cdist(source_points, target_points), axis=1)) + np.mean(np.min(distance.cdist(target_points, source_points), axis=1))

        # 3. Mean Square Error (MSE)
        mse[i,j]  = np.max(np.square(np.linalg.norm(source_points - target_points, axis=1)))

        # 4 Principal Component Analysis (PCA)
        source_pca = PCA(n_components=3).fit_transform(source_points)
        target_pca = PCA(n_components=3).fit_transform(target_points)
        pca_dist[i,j]  = np.mean(np.square(np.linalg.norm(source_pca - target_pca, axis=1)))


#Printing 
print('Hausdorff Distance')
print(np.max(hausdorff_dist))
print('Chamfer Distance')
print(np.max(chamfer_dist))
#print heat map of the distances
import seaborn as sns
plt.figure()
sns.heatmap(hausdorff_dist, annot=True, cmap='viridis')
plt.title('Hausdorff Distance', fontsize=20)

plt.figure()
sns.heatmap(chamfer_dist, annot=True, cmap='viridis')
plt.title('Chamfer Distance', fontsize=20)

plt.figure()
sns.heatmap(mse, annot=True, cmap='viridis')
plt.title('MSE', fontsize=20)

plt.figure()
sns.heatmap(pca_dist, annot=True, cmap='viridis')
plt.title('PCA', fontsize=20)
plt.show()

