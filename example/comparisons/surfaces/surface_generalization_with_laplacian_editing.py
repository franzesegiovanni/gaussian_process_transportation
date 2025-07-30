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

#%% Transport the dynamical system on the new surface
transport=Transport(training_traj=X)


print('Transporting the dynamical system on the new surface')
transport.fit(source_distribution=source_distribution, target_distribution=target_distribution, do_scale=False, do_rotation=True)
X_hat=transport.transport(X=X)
deltaX_hat=transport.transport_velocity(velocity=deltaX)


#%% Fit a dynamical system to the demo and plot it

plot_vector_field(X, deltaX, source_distribution, min_var=False)
# Fit the Gaussian Process dynamical system   
plot_vector_field(X_hat, deltaX_hat, target_distribution, min_var=False)
mask_traj=transport.mask_traj
mask_source=transport.mask_source
#plot connecting lines from target to trajectory according to mask_traj and mask_source
traj_connected=  X_hat[mask_traj]
target_connected= target_distribution[mask_source]
for i in range(len(traj_connected)):
    plt.plot([traj_connected[i,0], target_connected[i,0]], [traj_connected[i,1], target_connected[i,1]], 'k-', lw=1)
plt.show()