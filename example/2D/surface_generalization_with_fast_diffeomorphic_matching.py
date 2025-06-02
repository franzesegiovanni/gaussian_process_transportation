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
from policy_transportation.transportation.transportation import PolicyTransportation

from policy_transportation.models.locally_weighted_translations import Iterative_Locally_Weighted_Translations
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
X=resample(X, num_points=100)
source_distribution=resample(S, num_points=30)
target_distribution=resample(S1, num_points=30)

#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])

deltaX[-1,:]=X[0,:]-X[-1,:]


#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
plot_vector_field(gp_deltaX, x_grid,y_grid,X,source_distribution)

fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(source_distribution[:,0],source_distribution[:,1], color=[0,1,0])   
plt.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1]) 
plt.legend(["Demonstration","Surface","New Surface"])
#%% Transport the dynamical system on the new surface

transport=PolicyTransportation()

method = Iterative_Locally_Weighted_Translations(num_iterations=10, rho=0.3, beta=0.9)
transport.set_method(method=method, is_residual=False)

transport.fit(source_distribution, target_distribution, do_scale=False, do_rotation=True)

X_hat=transport.transport(X, return_std=False)
deltaX_hat=transport.transport_velocity(X, deltaX, return_var=False)
# quat_hat= transport.transport_orientation(X, quat)

# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2),  nu=2.5) + WhiteKernel(0.01)    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X_hat, deltaX_hat)
x1_grid=np.linspace(np.min(X_hat[:,0]-10), np.max(X_hat[:,0]+10), 100)
y1_grid=np.linspace(np.min(X_hat[:,1]-10), np.max(X_hat[:,1]+10), 100)
plot_vector_field(gp_deltaX1, x1_grid,y1_grid,X_hat,target_distribution )
plt.show()