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
from policy_transportation.models.rbf_regressor import RBFRegression
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from policy_transportation import GaussianProcess as GPR
from policy_transportation.models.locally_weighted_translations import Iterative_Locally_Weighted_Translations
import pathlib
import warnings

from policy_transportation.plot_utils import plot_orientation_frame
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore")

#%% Load the drawings

source_path = str(pathlib.Path(__file__).parent.absolute())  
# Load keypoints data
source_distribution = np.load(source_path + '/data/left_bag_keypoints.npy', allow_pickle=True)
target_distribution = np.load(source_path + '/data/left_bag_keypoints.npy', allow_pickle=True)

trajectory = np.load(source_path + '/data/left_bag_trajectory.npy', allow_pickle=True)

trajectory_validation = np.load(source_path + '/data/left_bag_trajectory.npy', allow_pickle=True)

position= trajectory[:, 1:4]  # Extract position data from the trajectory
orientation = trajectory[:, 4:8]  # Extract orientation data from the trajectory
time_original = np.arange(len(trajectory))  # Extract time or create indices

position_validation = trajectory_validation[:, 1:4]  # Extract position data from the validation trajectory
orientation_validation = trajectory_validation[:, 4:8]  # Extract orientation data from the validation trajectory
#%% Transport the dynamical system on the new surface

transport=PolicyTransportation()
# method = Iterative_Locally_Weighted_Translations(num_iterations=30, rho=1, beta=0.9)
# method = GPR(kernel=C(0.1) * RBF(length_scale=[0.3]) + WhiteKernel(0.0001))
method = RBFRegression( beta=20, sigma2=0.001) # large beta for smoother results, increase sigma2 when the data is noisy
transport.set_method(method=method, is_residual=method.is_residual)

transport.fit(source_distribution, target_distribution, do_scale=False, do_rotation=False)


position_hat=transport.transport(position, return_std=False)
oritentation_hat= transport.transport_orientation(position, orientation)


#%% Plot the results

fig = plt.figure(figsize=(12, 7))
# Create a 3D subplot for source and original trajectory
ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(source_distribution[:, 0], source_distribution[:, 1], source_distribution[:, 2], 
#            color='blue', alpha=0.5, label='Source Distribution')
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

#%% Plot Euler angles (roll, pitch, yaw) before and after transportation

# Convert quaternions to Euler angles
def quaternion_to_euler_scipy(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in radians using scipy.
    Args:
        q: quaternion array with shape (n, 4) where columns are [x, y, z, w]
    Returns:
        euler: array with shape (n, 3) where columns are [roll, pitch, yaw]
    """
    if q.ndim == 1:
        q = q.reshape(1, -1)
    euler = []
    for quat in q:
        # scipy expects [x, y, z, w], but you want [y, z, w, x]
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler.append(rot.as_euler('xyz', degrees=False))
    return np.array(euler)

euler_original = quaternion_to_euler_scipy(orientation)
euler_transported = quaternion_to_euler_scipy(oritentation_hat)

# Create time array for plotting
time_array = time_original

# Create a new figure for Euler angles
fig2, axes = plt.subplots(3, 1, figsize=(12, 10))

# Roll plot
axes[0].plot(time_array, np.degrees(euler_original[:, 0]), 'b-', linewidth=2, label='Original Roll')
axes[0].plot(time_array, np.degrees(euler_transported[:, 0]), 'r--', linewidth=2, label='Transported Roll')
axes[0].set_ylabel('Roll (degrees)')
axes[0].set_title('Roll Angle vs Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Pitch plot
axes[1].plot(time_array, np.degrees(euler_original[:, 1]), 'b-', linewidth=2, label='Original Pitch')
axes[1].plot(time_array, np.degrees(euler_transported[:, 1]), 'r--', linewidth=2, label='Transported Pitch')
axes[1].set_ylabel('Pitch (degrees)')
axes[1].set_title('Pitch Angle vs Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Yaw plot
axes[2].plot(time_array, np.degrees(euler_original[:, 2]), 'b-', linewidth=2, label='Original Yaw')
axes[2].plot(time_array, np.degrees(euler_transported[:, 2]), 'r--', linewidth=2, label='Transported Yaw')
axes[2].set_ylabel('Yaw (degrees)')
axes[2].set_xlabel('Time')
axes[2].set_title('Yaw Angle vs Time')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()