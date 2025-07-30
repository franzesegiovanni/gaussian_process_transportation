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
from policy_transportation.models.torch.ensemble_neural_network  import EnsembleNeuralNetwork
from policy_transportation import GaussianProcess as GPR    
from policy_transportation.transportation.transportation import PolicyTransportation


import pathlib
from policy_transportation.plot_utils import plot_vector_field
from policy_transportation.utils import resample
import warnings
warnings.filterwarnings("ignore")
#%% Load the drawings

source_path = str(pathlib.Path(__file__).parent.absolute())  
data =np.load(source_path+ '/data/'+str('example2')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
X=resample(X, num_points=200)
source_distribution=resample(S, num_points=20)
target_distribution=resample(S1, num_points=20)

#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])

#%% Transport the dynamical system on the new surface
k_transport = C(constant_value=10)  * RBF(1*np.ones(2), [0.1,5]) + WhiteKernel(0.01)


method = EnsembleNeuralNetwork(n_estimators=10)

transport=PolicyTransportation()

transport.set_method(method=method, is_residual=method.is_residual)

transport.fit(source_distribution, target_distribution, do_scale=False, do_rotation=True)
print('Transporting the dynamical system on the new surface')

X_hat = transport.transport(X, return_std=False)
deltaX_hat = transport.transport_velocity(X, deltaX, return_var=False)




#%% Fit a dynamical system to the demo and plot it
kernel_policy = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=kernel_policy)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
plot_vector_field(X, deltaX, source_distribution, min_var=False)
fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(source_distribution[:,0],source_distribution[:,1], color=[0,1,0])   
plt.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1]) 
plt.legend(["Demonstration","Surface","New Surface"])
# Fit the Gaussian Process dynamical system     
gp_deltaX_hat=GPR(kernel=kernel_policy)

gp_deltaX_hat.fit(X_hat, deltaX_hat)
x1_grid=np.linspace(np.min(X_hat[:,0]-10), np.max(X_hat[:,0]+10), 100)
y1_grid=np.linspace(np.min(X_hat[:,1]-10), np.max(X_hat[:,1]+10), 100)
plot_vector_field(X_hat, deltaX_hat, target_distribution, min_var=False)
plt.show()