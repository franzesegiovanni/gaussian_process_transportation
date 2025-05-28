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
    def __init__(self, kernel=C(0.1, constant_value_bounds=[0.1,2]) * RBF(length_scale=[0.1], length_scale_bounds=[0.05, 0.2]) + WhiteKernel(0.00001), do_scale=False, do_rotation=True):
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.movement_primitive=KMP()
        self.kernel= kernel
    def find_matching_waypoints(self, source_distribution, training_traj):

       # ceate cdist matrix
        distance_matrix = np.linalg.norm(training_traj[:, None] - source_distribution, axis=2)


        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        # mask= np.linalg.norm(training_traj[row_ind] - source_distribution[col_ind],axis=1) < 5
        # row_ind=row_ind[mask]
        # col_ind=col_ind[mask]

        return row_ind, col_ind
    def fit_transportation(self):
        
        self.affine_transform.fit(self.source_distribution, self.target_distribution)
 
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


    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=np.copy(self.training_traj)
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.training_traj, self.std= self.movement_primitive.predict(self.time, return_std=True)
 
        if hasattr(self, 'training_delta'):
            self.new_training_vel, self.vel_var = self.movement_primitive.derivative(self.time,return_var=True)
            # self.training_delta= self.new_training_vel * (self.time[1]-self.time[0])
            self.var_vel_transported = self.vel_var * (self.time[1]-self.time[0])**2
            # J = self.training_delta[:,:,np.newaxis] @ np.linalg.pinv(training_delta_old[:,:,np.newaxis])
            # J = (self.training_traj[1:,:,np.newaxis]- self.training_traj[:-1,:,np.newaxis]) @ np.linalg.pinv(self.training_traj_old[1:,:,np.newaxis]- self.training_traj_old[:-1,:,np.newaxis])
            # J = np.concatenate((J, J[-1:,:,:]), axis=0)
            # self.training_delta= J @ self.training_delta
            # self.training_delta = self.training_delta[:,:,0]
            self.training_delta[:-1,:]=(self.training_traj[1:,:]- self.training_traj[:-1,:])
        if hasattr(self, 'training_ori'):
            J = (self.training_traj[1:,:,np.newaxis]- self.training_traj[:-1,:,np.newaxis]) @ np.linalg.pinv(self.training_traj_old[1:,:,np.newaxis]- self.training_traj_old[:-1,:,np.newaxis])
            J = np.concatenate((J, J[-1:,:,:]), axis=0)
            quat_demo=quaternion.from_float_array(self.training_ori)
            quat_gp = quaternion.from_rotation_matrix(J, nonorthogonal=True)
            quat_transport=quat_gp * quat_demo
            self.training_ori= quaternion.as_float_array(quat_transport)

    def accuracy(self):
        traj_at_tstar= self.movement_primitive.predict(self.time_star)
        error=np.sqrt(np.mean((traj_at_tstar-self.y_star)**2))
        return error      

    def sample_transportation(self):
        training_traj_samples= self.movement_primitive.samples(self.traj_rotated)
        training_traj_samples= training_traj_samples.transpose(2,0,1)

        return training_traj_samples
