"""
Authors: Giovanni Franzese 
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation.models.torch.ensemble_bijective_network import EnsembleBijectiveNetwork as BiJectiveNetwork
from policy_transportation.transportation.policy_transportation import PolicyTransportation
class Neural_Transport():
    def __init__(self):
        super(Neural_Transport, self).__init__()
        self.method=PolicyTransportation(BiJectiveNetwork)

    
    def fit_transportation(self, do_scale=False, do_rotation=True):
        self.method.fit(self.source_distribution, self.target_distribution, do_scale=do_scale, do_rotation=do_rotation)


    def apply_transportation(self):
        self.training_traj, self.std=self.method.transport(self.training_traj)
        if hasattr(self, 'training_delta'):
            self.training_delta, self.var_vel_transported =self.method.transport_velocity(self.training_traj, self.training_delta)
        if hasattr(self, 'training_ori'):
            self.training_ori=self.method.transport_orientation(self.training_traj, self.training_ori)

    def sample_transportation(self):
        samples=self.method.sample_transportation(self.training_traj)
        return samples

