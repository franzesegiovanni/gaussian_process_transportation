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
class KMP_transportation():
    def __init__(self, kernel=C(0.1, constant_value_bounds=[0.1,2]) * RBF(length_scale=[0.1], length_scale_bounds=[0.05, 0.2]) + WhiteKernel(0.00001), do_scale=False, do_rotation=True):
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.transportation=KMP()
        self.kernel= kernel
    
    def fit_transportation(self):
        self.transportation.mask_traj, self.transportation.mask_dist= self.transportation.find_matching_waypoints(self.source_distribution, self.training_traj)
        
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        self.training_traj= self.affine_transform.predict(self.training_traj)
        
        self.transportation.fit(source_distribution, self.target_distribution, self.training_traj, kernel=self.kernel) 


    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.training_traj, self.std= self.transportation.predict(self.traj_rotated, return_std=True)
 
        if hasattr(self, 'training_delta'):
            J = (self.training_traj[1:,:,np.newaxis]- self.training_traj[:-1,:,np.newaxis]) @ np.linalg.pinv(self.training_traj_old[1:,:,np.newaxis]- self.training_traj_old[:-1,:,np.newaxis])
            J = np.concatenate((J, J[-1:,:,:]), axis=0)
            self.training_delta= (J @ self.training_delta[:,:,np.newaxis])[:,:,0]

    def sample_transportation(self):
        training_traj_samples= self.transportation.samples(self.traj_rotated)
        training_traj_samples= training_traj_samples.transpose(2,0,1)

        return training_traj_samples
