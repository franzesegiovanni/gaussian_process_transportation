"""
Authors: Giovanni Franzese
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation import AffineTransform
from policy_transportation.models.ensemble_random_forest import Ensemble_RF
class RFTrasportation():
    def __init__(self):
        super(RFTrasportation, self).__init__()

    def fit_transportation(self):
        self.affine_transform=AffineTransform(do_scale=False, do_rotation=True)
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        self.delta_distribution = self.target_distribution - source_distribution

        self.gp_delta_map=Ensemble_RF(n_estimators=50, max_depth=5)
     
        self.gp_delta_map.fit(source_distribution, self.delta_distribution)  


    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.delta_map_mean, self.std= self.gp_delta_map.predict(self.traj_rotated, return_std=True)

        self.training_traj = self.traj_rotated + self.delta_map_mean 

        for j in range(len(self.training_traj)-1):
            self.training_delta[j,:]=(self.training_traj[j+1,:]-self.training_traj[j,:])

    def sample_transportation(self):
        delta_map_samples= self.gp_delta_map.samples(self.traj_rotated)
        training_traj_samples = self.traj_rotated + delta_map_samples 
        return training_traj_samples
