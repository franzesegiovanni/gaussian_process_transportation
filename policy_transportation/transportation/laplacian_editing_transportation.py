"""
Authors: Giovanni Franzese
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import numpy as np 
from policy_transportation import AffineTransform
from policy_transportation.models.laplacian_editing import Laplacian_Editing
class LaplacianEditingTransportation():
    def __init__(self, training_traj):

        self.training_traj=training_traj
    def fit(self, source_distribution, target_distribution, do_scale=True, do_rotation=True):
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.method=Laplacian_Editing()
        self.affine_transform.fit(source_distribution, target_distribution)

        source_distribution_rotated=self.affine_transform.predict(source_distribution)  

        self.training_traj= self.affine_transform.predict(self.training_traj)

        self.method.fit(source_distribution_rotated, target_distribution, self.training_traj) 
        self.mask_traj=self.method.mask_traj
        self.mask_source=self.method.mask_source
        self.accuracy= self.method.accuracy

    def transport(self, X=None, return_std=False):

        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.training_traj, self.std= self.method.predict(self.traj_rotated, return_std=True)
        if return_std:
            return self.training_traj, self.std
        else:
            return self.training_traj

    def transport_velocity(self, position=None, velocity=None, return_var=False):

        J = (self.training_traj[1:,:,np.newaxis]- self.training_traj[:-1,:,np.newaxis]) @ np.linalg.pinv(self.training_traj_old[1:,:,np.newaxis]- self.training_traj_old[:-1,:,np.newaxis])
        J = np.concatenate((J, J[-1:,:,:]), axis=0)
        velocity_hat= (J @ velocity[:,:,np.newaxis])[:,:,0]
        return velocity_hat  


    def sample_transportation(self, X=None, n_samples=10):
        training_traj_samples= self.method.samples(self.traj_rotated, n_samples=n_samples)
        return training_traj_samples
