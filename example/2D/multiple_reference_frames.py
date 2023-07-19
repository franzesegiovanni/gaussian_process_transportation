"""
Authors: Ravi Prakash and Giovanni Franzese, Dec 2022
Email: r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
import pathlib
from plot_utils import plot_vector_field_minvar
from GILoSA import GaussianProcess as GPR
from GILoSA import Transport
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline

#%% Load the drawings

data =np.load(str(pathlib.Path().resolve())+'/data/'+str('demo')+'.npz')
X=data['demo'] 
S= X[(50,len(X)-1),:]+[[1,0],[-1,0]]
S1= S+[[-10,10],[-10,0]]
fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(S[:,0],S[:,1], color=[0,1,0])   
plt.scatter(S1[:,0],S1[:,1], color=[0,0,1]) 
plt.legend(["Demonstration","Source_distribution","Target distribution"])


#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])

## Downsample
X=X[::2,:]
deltaX=deltaX[::2,:]

#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
transport=Transport()
transport.kernel_transport=C(constant_value=np.sqrt(0.1))  * RBF(np.ones(1), length_scale_bounds=[10,100]) + WhiteKernel(0.0001, [0.001, 0.001 ])
transport.source_distribution=S 
transport.target_distribution=S1
transport.training_traj=X
transport.training_delta=deltaX
transport.fit_transportation()
transport.apply_transportation()
X1=transport.training_traj
deltaX1=transport.training_delta 

fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X1[:,0],X1[:,1], color=[1,0,0]) 
plt.scatter(S1[:,0],S1[:,1], color=[0,1,0])   
plt.scatter(X[:,0],X[:,1], color=[0,0,1]) 
plt.scatter(S[:,0],S[:,1], color=[0,0,0])   

x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 100)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 100)

k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)
plot_vector_field_minvar(gp_deltaX1, x1_grid,y1_grid,X1,S1)
plt.show()
