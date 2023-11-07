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
source_distribution=resample(S, num_points=100)
target_distribution=resample(S1, num_points=100)

fig = plt.figure(figsize = (12, 7))
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

x_lim=[np.min(X[:,0]-10), np.max(X[:,0]+20)]
y_lim=[np.min(X[:,1]-5), np.max(X[:,1]+30)]
## Plot diffeomorphism

fig = plt.figure(figsize = (12, 7))
num_points=15
x_grid=np.linspace(-40, 30, num_points)
y_grid=np.linspace(-30, 0, num_points)
X, Y = np.meshgrid(x_grid, y_grid)
#reshape it 

grid=np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))

plt.scatter(grid[:,0],grid[:,1], color=[1,0,0])

for i in range(num_points):
        plt.plot(X[i,:],Y[i,:], color=[0,0,0])
        plt.plot(X[:,i],Y[:,i], color=[0,0,0])
plt.scatter(source_distribution[:,0],source_distribution[:,1], color=[0,1,0])
plt.xlim(x_lim)
plt.ylim(y_lim)
# plt.axis('equal')
transport.training_traj=grid
# transport.training_delta=None
fig = plt.figure(figsize = (12, 7))
transport.apply_transportation()
grid_new=transport.training_traj
X=grid_new[:,0].reshape(num_points,num_points)
Y=grid_new[:,1].reshape(num_points,num_points)
# plt.contourf(X, Y, cmap='viridis', levels=20) 
plt.scatter(grid_new[:,0],grid_new[:,1], color=[1,0,0])


plt.xlim(x_lim)
plt.ylim(y_lim)
# plt.axis('equal')
for i in range(num_points):
        plt.plot(X[i,:],Y[i,:], color=[0,0,0])
        plt.plot(X[:,i],Y[:,i], color=[0,0,0])

plt.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,1,0])

plt.show()