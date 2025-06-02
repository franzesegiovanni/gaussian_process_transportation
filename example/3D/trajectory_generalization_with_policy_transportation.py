"""
Authors:  Giovanni Franzese and Ravi Prakash, Dec 2022
Email: g.franzese@tudelft.nl, r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from policy_transportation.transportation.transportation import PolicyTransportation
from policy_transportation.models.torch.bijective_neural_network import BiJectiveNetwork
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from policy_transportation import GaussianProcess as GPR
from policy_transportation.models.locally_weighted_translations import Iterative_Locally_Weighted_Translations
import pathlib
import warnings

from policy_transportation.plot_utils import plot_orientation_frame

warnings.filterwarnings("ignore")
#%% Load the drawings

source_path = str(pathlib.Path(__file__).parent.absolute())  
# data =np.load(source_path+ '/data/'+str('example2')+'.npz')
# Load keypoints data
source_distribution = np.load(source_path + '/data/source.npy', allow_pickle=True)
target_distribution = np.load(source_path + '/data/target.npy', allow_pickle=True)

trajectory = np.load(source_path + '/data/trajectory_1.npy', allow_pickle=True)

trajectory_validation = np.load(source_path + '/data/trajectory_2.npy', allow_pickle=True)

position= trajectory[:, 1:4]  # Extract position data from the trajectory
orientation = trajectory[:, 4:8]  # Extract orientation data from the trajectory

position_validation = trajectory_validation[:, 1:4]  # Extract position data from the validation trajectory
orientation_validation = trajectory_validation[:, 4:8]  # Extract orientation data from the validation trajectory
#%% Transport the dynamical system on the new surface

transport=PolicyTransportation()
method = Iterative_Locally_Weighted_Translations(num_iterations=10, rho=0.3, beta=0.9)
# method = GPR(kernel=C(0.1) * RBF(length_scale=[0.1]) + WhiteKernel(0.0001))
# method = BiJectiveNetwork(num_epochs=1000, num_blocks=4, num_hidden=50)
transport.set_method(method=method, is_residual=False)

transport.fit(source_distribution, target_distribution, do_scale=False, do_rotation=False)


position_hat=transport.transport(position, return_std=False)
oritentation_hat= transport.transport_orientation(position, orientation)


#%% Plot the results

fig = plt.figure(figsize=(12, 7))
# Create a 3D subplot for source and original trajectory
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(source_distribution[:, 0], source_distribution[:, 1], source_distribution[:, 2], 
           color='blue', alpha=0.5, label='Source Distribution')
ax1.scatter(position[:, 0], position[:, 1], position[:, 2], 
           color='black', s=15, label='Original Trajectory')
plot_orientation_frame(ax1, position, orientation)
ax1.set_title('Source Distribution and Original Trajectory')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# Create a 3D subplot for target and transported trajectory
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(target_distribution[:, 0], target_distribution[:, 1], target_distribution[:, 2], 
           color='blue', alpha=0.5, label='Target Distribution')
ax2.scatter(position_hat[:, 0], position_hat[:, 1], position_hat[:, 2], 
           color='black', s=15, label='Transported Trajectory')
ax2.scatter(position_validation[:, 0], position_validation[:, 1], position_validation[:, 2], 
           color='green', s=15, label='Validation Trajectory')
plot_orientation_frame(ax2, position_hat, oritentation_hat)
ax2.set_title('Target Distribution and Transported Trajectory')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()

plt.tight_layout()
plt.show()