"""
Authors:  Giovanni Franzese and Ravi Prakash, Dec 2022
Email: g.franzese@tudelft.nl, r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from policy_transportation import PolicyTransportation
from policy_transportation.models.rbf_regressor import RBFRegression
import pathlib
from policy_transportation.utils import resample
import warnings
from policy_transportation.plot_utils import plot_vector_field
warnings.filterwarnings("ignore")
#%% Load the drawings

source_path = str(pathlib.Path(__file__).parent.absolute())  
data =np.load(source_path+ '/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
X=resample(X, num_points=100)
source_distribution=resample(S, num_points=20)
target_distribution=resample(S1, num_points=20)

#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])


transport=PolicyTransportation()

method = RBFRegression( beta=20, sigma2=0.001)
transport.set_method(method=method, is_residual=method.is_residual)
    
transport.fit(source_distribution, target_distribution, do_scale=False, do_rotation=True)

X_hat=transport.transport(X, return_std=False)
deltaX_hat=transport.transport_velocity(X, deltaX, return_var=False)
# plt.plot(source_distribution[:,0], source_distribution[:,1], 'ro')
# Fit the Gaussian Process dynamical system
plot_vector_field(X, deltaX, source_distribution, min_var=False)
plot_vector_field(X_hat, deltaX_hat, target_distribution, min_var=False)
plt.show()
