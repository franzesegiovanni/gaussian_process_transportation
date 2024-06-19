"""
Authors: Giovanni Franzese, June 2024
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""


import numpy as np
import matplotlib.pyplot as plt
import pathlib
from policy_transportation.plot_utils import plot_traj_3D
from policy_transportation.transportation.torch.stocastic_variational_gaussian_process_transportation import SVGPTransport as Transport
import warnings
from matplotlib import cm
warnings.filterwarnings("ignore")
# Load the drawings

source_path = str(pathlib.Path(__file__).parent.parent.absolute())  
data =np.load(source_path+ '/data/'+str('example')+'.npz')
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

plot_traj_3D(X,S)
source_distribution =S.reshape(-1,3)  
target_distribution =S1.reshape(-1,3)

print("Fit Gaussian Process Tranportation")
#%% Transport the dynamical system on the new surface
transport=Transport()
transport.source_distribution=source_distribution 
transport.target_distribution=target_distribution
transport.training_traj=X
transport.fit_transportation(num_epochs=10,num_inducing=100)
transport.apply_transportation()

X1=transport.training_traj

plot_traj_3D(X1,S1)

plt.show()
