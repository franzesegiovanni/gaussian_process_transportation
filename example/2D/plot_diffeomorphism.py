"""
Authors:  Giovanni Franzese and Ravi Prakash, Dec 2022
Email: g.franzese@tudelft.nl, r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from policy_transportation import GaussianProcess as GPR
from policy_transportation import GaussianProcessTransportation as Transport
import pathlib
from policy_transportation.plot_utils import plot_vector_field 
from policy_transportation.utils import resample
import warnings
warnings.filterwarnings("ignore")
#%% Load the drawings

source_path = str(pathlib.Path(__file__).parent.absolute())  
data =np.load(source_path+ '/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
source_distribution=resample(S, num_points=20)
target_distribution=resample(S1, num_points=20)

fig = plt.figure()
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(source_distribution[:,0],source_distribution[:,1], color=[0,1,0])   
plt.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1]) 
plt.legend(["Demonstration","Surface","New Surface"])
#%% Transport the dynamical system on the new surface
transport=Transport()
transport.source_distribution=source_distribution 
transport.target_distribution=target_distribution

k_transport = C(constant_value=np.sqrt(0.1))  * RBF(40*np.ones(2), length_scale_bounds=[0.01, 500]) + WhiteKernel(0.01, noise_level_bounds=[0.01, 0.1] )
transport.kernel_transport=k_transport
print('Transporting the dynamical system on the new surface')
transport.fit_transportation()

x_lim=[np.min(X[:,0]-15), np.max(X[:,0]+15)]
y_lim=[np.min(X[:,1]-15), np.max(X[:,1]+15)]
## Plot diffeomorphism

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

axs[0].scatter(source_distribution[:, 0], source_distribution[:, 1], color='red', label='Source Distribution')
axs[0].scatter(target_distribution[:, 0], target_distribution[:, 1], color='blue', label='Target Distribution')
axs[0].set_xlim(x_lim)
axs[0].set_ylim(y_lim)
axs[0].set_title('Distribution Match', fontsize=20)
axs[0].legend()
# Plot connecting lines between each of the points
for i in range(len(source_distribution)):
        axs[0].plot([source_distribution[i, 0], target_distribution[i, 0]], [source_distribution[i, 1], target_distribution[i, 1]], color='black', alpha=0.5, linewidth=0.5)

num_points=15
x_grid=np.linspace(-40, 30, num_points)
y_grid=np.linspace(-30, 0, num_points)
X, Y = np.meshgrid(x_grid, y_grid)
#reshape it 

grid=np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))

norms = np.linalg.norm(grid - np.min(grid, axis=0), axis=1)
colors = norms[:, np.newaxis] / norms.max()  # normalize the norms to [0, 1]
cmap='PiYG'
axs[1].scatter(grid[:,0],grid[:,1], c=colors, cmap=cmap)
axs[1].legend()
for i in range(num_points):
        axs[1].plot(X[i,:],Y[i,:], color=[0,0,0])
        axs[1].plot(X[:,i],Y[:,i], color=[0,0,0])
axs[1].scatter(source_distribution[:,0],source_distribution[:,1], color=[1,0,0], label='Source Distribution')
axs[1].set_xlim(x_lim)
axs[1].set_ylim(y_lim)
axs[1].set_title('Source distribution', fontsize=20)
axs[1].set_yticks([])
axs[1].set_yticklabels([])
axs[1].legend()
# plt.axis('equal')

transport.training_traj=grid
# transport.training_delta=None

transport.apply_transportation_linear()
grid_new=transport.training_traj

transport.training_traj=source_distribution
transport.apply_transportation_linear()
source_distribution_new=transport.training_traj

X=grid_new[:,0].reshape(num_points,num_points)
Y=grid_new[:,1].reshape(num_points,num_points)
# plt.contourf(X, Y, cmap=cmap, levels=20) 
axs[2].scatter(grid_new[:,0],grid_new[:,1],c=colors, cmap=cmap)


axs[2].set_xlim(x_lim)
axs[2].set_ylim(y_lim)
# plt.axis('equal')
for i in range(num_points):
        axs[2].plot(X[i,:],Y[i,:], color=[0,0,0])
        axs[2].plot(X[:,i],Y[:,i], color=[0,0,0])

axs[2].scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1], label='Target Distribution')
axs[2].scatter(source_distribution_new[:,0], source_distribution_new[:,1], facecolors='none', edgecolors=[1,0,0], linewidths=2,label='Source Distribution')
axs[2].set_title('Affine Transformation', fontsize=20)
axs[2].set_yticks([])
axs[2].set_yticklabels([])
transport.training_traj=grid
# transport.training_delta=None
# transport.optimize_diffeomorphism()
transport.apply_transportation()
grid_new=transport.training_traj

transport.training_traj=source_distribution
transport.apply_transportation()
source_distribution_new=transport.training_traj
X=grid_new[:,0].reshape(num_points,num_points)
Y=grid_new[:,1].reshape(num_points,num_points)
axs[2].legend()

# plt.contourf(X, Y, cmap=cmap, levels=20) 

axs[3].scatter(grid_new[:,0],grid_new[:,1], c=colors, cmap=cmap)


axs[3].set_xlim(x_lim)
axs[3].set_ylim(y_lim)
# plt.axis('equal')
for i in range(num_points):
        axs[3].plot(X[i,:],Y[i,:], color=[0,0,0])
        axs[3].plot(X[:,i],Y[:,i], color=[0,0,0])

axs[3].scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1], label='Target Distribution')
axs[3].scatter(source_distribution_new[:,0], source_distribution_new[:,1], facecolors='none', edgecolors=[1,0,0], linewidths=2,label='Source Distribution')
axs[3].set_title(' GP Transportation', fontsize=20)
axs[3].legend()
axs[3].set_yticks([])
axs[3].set_yticklabels([])

plt.subplots_adjust(wspace=0.05)
plt.savefig(source_path+'/pictures/diffeomorphism.pdf', dpi=300, bbox_inches='tight')
plt.show()
