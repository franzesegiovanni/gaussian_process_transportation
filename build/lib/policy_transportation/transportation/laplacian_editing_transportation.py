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
    def __init__(self):
        super(LaplacianEditingTransportation, self).__init__()

    
    def fit_transportation(self,  do_scale=True, do_rotation=True):
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.transportation=Laplacian_Editing()
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        self.training_traj= self.affine_transform.predict(self.training_traj)
        
        self.transportation.fit(source_distribution, self.target_distribution, self.training_traj) 
        self.mask_traj=self.transportation.mask_traj
        self.mask_source=self.transportation.mask_source


    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.training_traj, self.std= self.transportation.predict(self.traj_rotated, return_std=True)


        if hasattr(self, 'training_delta'):
            J = (self.training_traj[1:,:,np.newaxis]- self.training_traj[:-1,:,np.newaxis]) @ np.linalg.pinv(self.training_traj_old[1:,:,np.newaxis]- self.training_traj_old[:-1,:,np.newaxis])
            J = np.concatenate((J, J[-1:,:,:]), axis=0)
            self.training_delta= (J @ self.training_delta[:,:,np.newaxis])[:,:,0]

    def accuracy(self):
        return self.transportation.accuracy    


    def sample_transportation(self):
        training_traj_samples= self.transportation.samples(self.traj_rotated)
        return training_traj_samples
