#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from policy_transportation import GaussianProcess as GPR
from policy_transportation.transportation.laplacian_editing_transportation import LaplacianEditingTransportation as Transport
from policy_transportation.plot_utils import plot_vector_field 
from policy_transportation.utils import resample
import warnings
import os
warnings.filterwarnings("ignore")
#%% Load the drawings

script_path = str(os.path.dirname(__file__))
data =np.load(script_path+'/data/'+str('example0')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
X=resample(X, num_points=200)
source_distribution=resample(S, num_points=100)
target_distribution=resample(S1, num_points=100)

#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])

#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
plot_vector_field(gp_deltaX, x_grid,y_grid,X,S)

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
transport.training_traj=X
transport.training_delta=deltaX

print('Transporting the dynamical system on the new surface')
transport.fit_transportation(do_scale=False, do_rotation=True)
transport.apply_transportation()
X1=transport.training_traj
deltaX1=transport.training_delta 

# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01 )    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)
x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 200)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 200)
plot_vector_field(gp_deltaX1, x1_grid,y1_grid,X1,target_distribution)
mask_traj=transport.mask_traj
mask_source=transport.mask_source
#plot connecting lines from target to trajectory according to mask_traj and mask_source
traj_connected=  X1[mask_traj]
target_connected= target_distribution[mask_source]
for i in range(len(traj_connected)):
    plt.plot([traj_connected[i,0], target_connected[i,0]], [traj_connected[i,1], target_connected[i,1]], 'k-', lw=1)
plt.show()