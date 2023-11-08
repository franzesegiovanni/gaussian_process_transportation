"""
Authors: Giovanni Franzese
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import numpy as np 
from policy_transportation import AffineTransform
from policy_transportation.models.kernelized_movemement_primitives import KMP
class KMP_transportation():
    def __init__(self, do_scale=False, do_rotation=True, treshold_distance=5.0 ):
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.transportation=KMP(treshold_distance=treshold_distance)
    
    def fit_transportation(self):
        self.transportation.find_matching_waypoints(self.source_distribution, self.training_traj)
        
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        self.training_traj= self.affine_transform.predict(self.training_traj)
        
        self.transportation.fit(source_distribution, self.target_distribution, self.training_traj) 


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
