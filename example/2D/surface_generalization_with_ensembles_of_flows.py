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
from policy_transportation.models.torch.ensemble_bijective_network import EnsembleBijectiveNetwork
from policy_transportation.models.torch.bijective_neural_network import BiJectiveNetwork
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
transport=PolicyTransportation()
# method = BiJectiveNetwork(num_epochs=200)
method = EnsembleBijectiveNetwork(num_epochs=200, n_estimators=10)
transport.set_method(method=method, is_residual=method.is_residual)
    
transport.fit(source_distribution, target_distribution, do_scale=False, do_rotation=True)

X_hat=transport.transport(X, return_std=False)
deltaX_hat=transport.transport_velocity(X, deltaX, return_var=False)

# Fit the Gaussian Process dynamical system
plot_vector_field(X, deltaX, source_distribution, min_var=False)
plot_vector_field(X_hat, deltaX_hat, target_distribution, min_var=False)
plt.show()
