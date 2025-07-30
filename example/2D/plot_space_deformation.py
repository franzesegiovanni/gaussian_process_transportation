"""
Authors:  Giovanni Franzese and Ravi Prakash, Dec 2022
Email: g.franzese@tudelft.nl, r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from policy_transportation import PolicyTransportation
from policy_transportation import GaussianProcess as GPR
import pathlib
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



x_lim=[np.min(X[:,0]-15), np.max(X[:,0]+15)]
y_lim=[np.min(X[:,1]-15), np.max(X[:,1]+15)]

num_points=15
x_grid=np.linspace(-40, 30, num_points)
y_grid=np.linspace(-30, 0, num_points)
X, Y = np.meshgrid(x_grid, y_grid)
grid=np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))



transport_affine=PolicyTransportation()
transport_affine.fit(source_distribution, target_distribution)

grid_affine=transport_affine.transport(grid, return_std=False)
X_affine=grid_affine[:,0].reshape(num_points,num_points)
Y_affine=grid_affine[:,1].reshape(num_points,num_points)
source_distribution_affine=transport_affine.transport(source_distribution, return_std=False)

transport=PolicyTransportation()
method = GPR(C(constant_value=10) * RBF(1*np.ones(2), [0.1,5]) + WhiteKernel(0.01))
transport.set_method(method=method, is_residual=method.is_residual)
transport.fit(source_distribution, target_distribution, do_scale=False, do_rotation=True)
grid_transport=transport.transport(grid, return_std=False)
X_transport=grid_transport[:,0].reshape(num_points,num_points)
Y_transport=grid_transport[:,1].reshape(num_points,num_points)
source_distribution_transported=transport.transport(source_distribution, return_std=False)



fig, axs = plt.subplots(1, 4, figsize=(16, 4))

axs[0].scatter(source_distribution[:, 0], source_distribution[:, 1], color='green', label='Source Distribution')
axs[0].scatter(target_distribution[:, 0], target_distribution[:, 1], color='blue', label='Target Distribution')
axs[0].set_xlim(x_lim)
axs[0].set_ylim(y_lim)
axs[0].set_title('Distribution Match', fontsize=16)
axs[0].legend()
# Plot connecting lines between each of the points
for i in range(len(source_distribution)):
        axs[0].plot([source_distribution[i, 0], target_distribution[i, 0]], [source_distribution[i, 1], target_distribution[i, 1]], color='black', alpha=0.5, linewidth=0.5)
norms = np.linalg.norm(grid - np.min(grid, axis=0), axis=1)
colors = norms[:, np.newaxis] / norms.max()  # normalize the norms to [0, 1]
cmap='cool'

axs[1].scatter(grid[:,0],grid[:,1], c=colors, cmap=cmap, alpha=0.3)
axs[1].legend()
for i in range(num_points):
        axs[1].plot(X[i,:],Y[i,:], color=[0,0,0])
        axs[1].plot(X[:,i],Y[:,i], color=[0,0,0])
axs[1].scatter(source_distribution[:,0],source_distribution[:,1], color='green', label='Source Distribution')
axs[1].set_xlim(x_lim)
axs[1].set_ylim(y_lim)
axs[1].set_title(' Position Labels ', fontsize=16)
axs[1].set_yticks([])
axs[1].set_yticklabels([])
axs[1].legend()
# plt.axis('equal')


axs[2].scatter(grid_affine[:,0],grid_affine[:,1],c=colors, cmap=cmap, alpha=0.3)
axs[2].set_xlim(x_lim)
axs[2].set_ylim(y_lim)
# plt.axis('equal')
for i in range(num_points):
        axs[2].plot(X_affine[i,:],Y_affine[i,:], color=[0,0,0])
        axs[2].plot(X_affine[:,i],Y_affine[:,i], color=[0,0,0])

axs[2].scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1], label='Target Distribution')
axs[2].scatter(source_distribution_affine[:,0], source_distribution_affine[:,1], facecolors='none', edgecolors='green', linewidths=2,label='Source Distribution')
axs[2].set_title('Linear Transformation', fontsize=16)
axs[2].set_yticks([])
axs[2].set_yticklabels([])
axs[2].legend()

axs[3].scatter(grid_transport[:,0],grid_transport[:,1], c=colors, cmap=cmap, alpha=0.3)
axs[3].set_xlim(x_lim)
axs[3].set_ylim(y_lim)
# plt.axis('equal')
for i in range(num_points):
        axs[3].plot(X_transport[i,:],Y_transport[i,:], color=[0,0,0])
        axs[3].plot(X_transport[:,i],Y_transport[:,i], color=[0,0,0])

axs[3].scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1], label='Target Distribution')
axs[3].scatter(source_distribution_transported[:,0], source_distribution_transported[:,1], facecolors='none', edgecolors='green', linewidths=2,label='Source Distribution')
axs[3].set_title('Nonlinear Transformation', fontsize=16)
axs[3].legend()
axs[3].set_yticks([])
axs[3].set_yticklabels([])

plt.subplots_adjust(wspace=0.05)
plt.savefig(source_path+'/pictures/space_deformation.pdf', dpi=300, bbox_inches='tight')
plt.show()
