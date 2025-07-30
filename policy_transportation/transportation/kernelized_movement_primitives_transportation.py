"""
Authors: Giovanni Franzese
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import numpy as np 
from policy_transportation import AffineTransform
from policy_transportation.models.kernelized_movemement_primitives import KMP
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import quaternion
from scipy.optimize import linear_sum_assignment
class KMP_transportation():
    def __init__(self, kernel=C(0.1, constant_value_bounds=[0.1,2]) * RBF(length_scale=[0.1], length_scale_bounds=[0.05, 0.2]) + WhiteKernel(0.00001), training_traj=None):
 
        self.movement_primitive=KMP()
        self.kernel= kernel
        self.training_traj=training_traj
    def find_matching_waypoints(self, source_distribution, training_traj):

       # ceate cdist matrix
        distance_matrix = np.linalg.norm(training_traj[:, None] - source_distribution, axis=2)


        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        return row_ind, col_ind
    def fit(self, source_distribution, target_distribution, do_scale=True, do_rotation=True):
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.source_distribution=source_distribution
        self.target_distribution=target_distribution
        self.affine_transform.fit(source_distribution, target_distribution)

        self.training_traj= self.affine_transform.predict(self.training_traj)
        self.time=np.linspace(0,1, self.training_traj.shape[0]).reshape(-1,1)
        self.movement_primitive.fit(self.time, self.training_traj, kernel=self.kernel) 

        source_distribution=self.affine_transform.predict(self.source_distribution)  
        mask_traj, mask_source= self.find_matching_waypoints(source_distribution, self.training_traj)
        diff= self.target_distribution[mask_source] - source_distribution[mask_source]
        self.time_star = self.time[mask_traj]
        self.y_star = self.training_traj[mask_traj] + diff
        self.movement_primitive.correct( self.time_star, self.y_star )
        self.mask_traj=mask_traj
        self.mask_source=mask_source
        traj_at_tstar= self.movement_primitive.predict(self.time_star)
        self.accuracy=np.sqrt(np.mean((traj_at_tstar-self.y_star)**2))
    def transport(self, X,return_std=False):
        if return_std:
            position_hat, std= self.movement_primitive.predict(self.time, return_std=True)
            return position_hat, std
        else:
            position_hat, std= self.movement_primitive.predict(self.time, return_std=False)
            return position_hat

    def transport_velocity(self, position, velocity, return_var=False):
        if return_var:
            vel, vel_var = self.movement_primitive.derivative(self.time,return_var=True)
            vel = np.squeeze(vel, axis=-1)
            vel_var = np.squeeze(vel_var, axis=-1)
            return  vel, vel_var
        else:
            vel = self.movement_primitive.derivative(self.time, return_var=False)
            vel = np.squeeze(vel, axis=-1)
            return  vel

    def sample_transportation(self, X=None, n_samples=10):
        training_traj_samples= self.movement_primitive.samples(X, n_samples=n_samples)
        training_traj_samples= training_traj_samples.transpose(2,0,1)

        return training_traj_samples
