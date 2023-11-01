"""
Authors:  Giovanni Franzese 
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from policy_transportation import GaussianProcess as GPR
from policy_transportation.transportation.torch.ensemble_bijective_transport import Neural_Transport as Transport
import pathlib
from policy_transportation.plot_utils import plot_vector_field 
from policy_transportation.utils import resample
import warnings
warnings.filterwarnings("ignore")
#%% Load the drawings

source_path = str(pathlib.Path(__file__).parent.parent.absolute())  
data =np.load(source_path+ '/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']

X=resample(X, num_points=200)
source_distribution=resample(S)
target_distribution=resample(S1)


fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(S[:,0],S[:,1], color=[0,1,0])   
plt.scatter(S1[:,0],S1[:,1], color=[0,0,1]) 
plt.legend(["Demonstration","Surface","New Surface"])


#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])

#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-25), np.max(X[:,0]+25), 100)
y_grid=np.linspace(np.min(X[:,1]-25), np.max(X[:,1]+25), 100)
plot_vector_field(gp_deltaX, x_grid,y_grid,X,S)
plt.xlim(np.min(X[:,0]-25), np.max(X[:,0]+25))
plt.ylim(np.min(X[:,1]-25), np.max(X[:,1]+25))

#%% Transport the dynamical system on the new surface
transport=Transport()
transport.source_distribution=source_distribution 
transport.target_distribution=target_distribution
transport.training_traj=X
transport.training_delta=deltaX
transport.fit_transportation(num_epochs=500)
transport.apply_transportation()
X1=transport.training_traj
deltaX1=transport.training_delta 
x1_grid=np.linspace(np.min(X[:,0]-25), np.max(X[:,0]+25), 100)
y1_grid=np.linspace(np.min(X[:,1]-25), np.max(X[:,1]+25), 100)

# Fit the Gaussian Process dynamical system   
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX1=GPR(kernel=k_deltaX1)
# print(X1)
mask = ~np.any(np.isnan(X1), axis=1)
gp_deltaX1.fit(X1[mask], deltaX1[mask])
plot_vector_field(gp_deltaX1, x1_grid,y1_grid,X1,S1)
plt.xlim(np.min(X[:,0]-25), np.max(X[:,0]+25))
plt.ylim(np.min(X[:,1]-25), np.max(X[:,1]+25))
plt.show()
