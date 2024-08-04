"""
Authors:  Giovanni Franzese and Ravi Prakash, Dec 2022
Email: g.franzese@tudelft.nl, r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from policy_transportation.models.coherent_point_drift import DeformableRegistration as Transport
import pathlib
from policy_transportation.plot_utils import plot_vector_field, plot_vector_field_minvar
from policy_transportation.utils import resample
import warnings
warnings.filterwarnings("ignore")
#%% Load the drawings

source_path = str(pathlib.Path(__file__).parent.absolute())  
data =np.load(source_path+ '/data/'+str('example4')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
X=resample(X, num_points=100)
source_distribution=resample(S, num_points=20)
target_distribution=resample(S1, num_points=20)

transport= Transport(source_distribution=source_distribution, target_distribution=target_distribution, beta=20, sigma2=0.001)
transport.fit()

source_transported = transport.predict(source_distribution)
demonstration_transported = transport.predict(X)

# plt.plot(source_distribution[:,0], source_distribution[:,1], 'ro')
plt.plot(source_transported[:,0], source_transported[:,1], 'bo')
plt.plot(target_distribution[:,0], target_distribution[:,1], 'go')
plt.plot(demonstration_transported[:,0], demonstration_transported[:,1], 'k-o')
plt.show()

