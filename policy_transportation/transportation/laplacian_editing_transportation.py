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
        self.affine_transform=AffineTransform(do_scale=False, do_rotation=True)
        self.transportation=Laplacian_Editing()
    
    def fit_transportation(self, threshold_distance = 5.0):
        
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        self.training_traj= self.affine_transform.predict(self.training_traj)
        
        self.transportation.fit(source_distribution, self.target_distribution, self.training_traj, threshold_distance=threshold_distance) 


    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.training_traj, self.std= self.transportation.predict(self.traj_rotated, return_std=True)
 
        self.training_delta=np.zeros_like(self.training_traj)
        for j in range(len(self.training_traj)-1):
            self.training_delta[j,:]=(self.training_traj[j+1,:]-self.training_traj[j,:])

    def sample_transportation(self):
        training_traj_samples= self.transportation.samples(self.traj_rotated)
        return training_traj_samples
