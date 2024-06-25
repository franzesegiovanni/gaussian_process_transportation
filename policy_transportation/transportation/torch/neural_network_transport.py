"""
Authors: Giovanni Franzese & Ravi Prakash, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation import AffineTransform
# from policy_transportation.models.torch.neural_network import NeuralNetwork
from policy_transportation.models.torch.neural_network import NeuralNetwork 
import pickle
import numpy as np
import quaternion
class Neural_Transport():
    def __init__(self):
        super(Neural_Transport, self).__init__()


    def fit_transportation(self, num_epochs=20):
        if type(self.target_distribution) != type(self.source_distribution):
            raise TypeError("Both the distribution must be a numpy array.")
        elif not(isinstance(self.target_distribution, np.ndarray)) and not(isinstance(self.source_distribution, np.ndarray)):
            self.convert_distribution_to_array() #this needs to be a function of every sensor class

        self.affine_transform=AffineTransform()
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        delta_distribution = self.target_distribution - source_distribution

        self.delta_map=NeuralNetwork(source_distribution, delta_distribution)
        self.delta_map.fit(source_distribution, delta_distribution, num_epochs=num_epochs)  

    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.delta_map_mean, self.std= self.gp_delta_map.predict(self.traj_rotated, return_std=True)
        # convert to numpyu array
        self.delta_map_mean=self.delta_map_mean.detach().cpu().numpy()
        self.std=self.std.detach().cpu().numpy()
        self.training_traj = self.traj_rotated + self.delta_map_mean 

        #Deform Deltas and orientation
        if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
            pos=(np.array(self.traj_rotated))
            Jacobian, Jacobian_std=self.gp_delta_map.derivative(pos)
            # convert the Jacobian and Jacobian_std to numpy
            Jacobian=Jacobian.detach().cpu().numpy()
            Jacobian_std=Jacobian_std.detach().cpu().numpy()

            rot_gp= np.eye(Jacobian[0].shape[0]) + Jacobian
            rot_affine= self.affine_transform.rotation_matrix
            derivative_affine= self.affine_transform.derivative(pos)


        if  hasattr(self, 'training_delta'):
            self.training_delta = self.training_delta[:,:,np.newaxis]

            self.training_delta=  derivative_affine @ self.training_delta
            self.var_vel_transported=Jacobian_std**2 @ self.training_delta**2

            self.training_delta= rot_gp @ self.training_delta
            self.training_delta=self.training_delta[:,:,0]
            self.var_vel_transported=self.var_vel_transported[:,:,0]


        if  hasattr(self, 'training_ori'):   
            quat_demo=quaternion.from_float_array(self.training_ori)
            quat_affine= quaternion.from_rotation_matrix(rot_affine)
            quat_gp = quaternion.from_rotation_matrix(rot_gp, nonorthogonal=True)
            quat_transport=quat_gp * (quat_affine * quat_demo)
            self.training_ori= quaternion.as_float_array(quat_transport)


    def sample_transportation(self):
        training_traj_samples= self.gp_delta_map.samples(self.traj_rotated)
        return training_traj_samples

