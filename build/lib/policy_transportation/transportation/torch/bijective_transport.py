"""
Authors: Giovanni Franzese 
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation import AffineTransform
from policy_transportation.models.torch.bijective_neural_network import BiJectiveNetwork
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
 
        self.gp_delta_map=BiJectiveNetwork(source_distribution, self.target_distribution)

        self.gp_delta_map.fit(num_epochs=num_epochs)  

    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.training_traj= self.gp_delta_map.predict(self.traj_rotated)

        #Deform Deltas and orientation
        if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
            pos=(np.array(self.traj_rotated))
            Jacobian=self.gp_delta_map.derivative(pos)
            rot_gp=  Jacobian
            derivative_affine= self.affine_transform.derivative(pos)
            J_phi= rot_gp @ derivative_affine

            print("Is the map locally diffeomorphic?", np.all(np.linalg.det(J_phi)) > 0)
            print(np.linalg.det(J_phi))
            print("percerntagle of non-diffeomorphic points", np.sum(np.linalg.det(J_phi)<=0)/len(J_phi))


        if  hasattr(self, 'training_delta'):
            self.training_delta = self.training_delta[:,:,np.newaxis]

            self.training_delta=  derivative_affine @ self.training_delta

            self.training_delta= rot_gp @ self.training_delta
            self.training_delta=self.training_delta[:,:,0]


    def sample_transportation(self):
        training_traj_samples= self.gp_delta_map.samples(self.traj_rotated)
        return training_traj_samples
