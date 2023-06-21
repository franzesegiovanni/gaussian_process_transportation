"""
Authors: Giovanni Franzese and Ravi Prakash, June 2023
Email: r.prakash-1@tudelft.nl, g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""


import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from GILoSA import GaussianProcess as GPR
import pathlib
from plot_utils import *
from GILoSA import Transport
import warnings
warnings.filterwarnings("ignore")
# Load the drawings

data =np.load(str(pathlib.Path().resolve())+'/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['old_surface'] 
S1=data['new_surface']
print(S.shape)

fig = plt.figure()
ax = plt.axes(projection ='3d')
newsurf = ax.plot_surface(S1[:,:,0], S1[:,:,1], S1[:,:,2], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf = ax.plot_surface(S[:,:,0], S[:,:,1], S[:,:,2], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(X[:,0], X[:,1],X[:,2], color='red') # parabola points

#Learn the dynamical system
deltaX = np.zeros((len(X),3))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(3),  nu=1.5) + WhiteKernel(0.01 )  
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)


x_grid=np.linspace(np.min(X[:,0]), np.max(X[:,0]), 10)
y_grid=np.linspace(np.min(X[:,1]), np.max(X[:,1]), 10)
z_grid=np.linspace(np.min(X[:,2]), np.max(X[:,2]), 3)
plot_traj_evolution(gp_deltaX,x_grid,y_grid,z_grid,X,S)
source_distribution =S.reshape(-1,3)  
target_distribution =S1.reshape(-1,3)

#%% Transport the dynamical system on the new surface
transport=Transport()
transport.source_distribution=source_distribution 
transport.target_distribution=target_distribution
transport.training_traj=X
transport.training_delta=deltaX
transport.fit_trasportation()
transport.apply_trasportation()
X1=transport.training_traj
deltaX1=transport.training_delta 

x1_grid=np.linspace(np.min(X[:,0]), np.max(X[:,0]), 10)
y1_grid=np.linspace(np.min(X[:,1]), np.max(X[:,1]), 10)
z1_grid=np.linspace(np.min(X[:,2]), np.max(X[:,2]), 3)

# Fit the Gaussian Process dynamical system   
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(3), nu=1.5) + WhiteKernel(0.01)   
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)
plot_traj_evolution(gp_deltaX1,x1_grid,y1_grid,z1_grid,X1,S1)

plt.show()
