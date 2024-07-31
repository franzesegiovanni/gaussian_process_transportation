"""
Authors: Giovanni Franzese & Ravi Prakash, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import optuna
from policy_transportation import AffineTransform
from policy_transportation.models.locally_weighted_translations import Iterative_Locally_Weighted_Translations
import numpy as np
import quaternion
import matplotlib.pyplot as plt
class DiffeomorphicTransportation():
    def __init__(self, num_iterations=30):
        super(DiffeomorphicTransportation, self).__init__()
        self.num_iterations=num_iterations

    def fit_transportation(self,do_scale=False, do_rotation=True):
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  

        self.gp_delta_map=Iterative_Locally_Weighted_Translations(para=[self.num_iterations, 1, 0.9])

        self.gp_delta_map.fit(source=source_distribution, target=self.target_distribution)  


    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.map_mean, self.std= self.gp_delta_map.predict(self.traj_rotated, return_std=True)
        self.training_traj = self.map_mean 

        #Deform Deltas and orientation
        if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
            pos=(np.array(self.traj_rotated))
            Jacobian, Jacobain_var=self.gp_delta_map.derivative(pos, return_var=True)

            rot_gp= Jacobian
            rot_affine= self.affine_transform.rotation_matrix
            derivative_affine= self.affine_transform.derivative(pos)

            J_phi = rot_gp @ derivative_affine
            print("Is the map diffeomorphic?", np.all((np.linalg.det(Jacobian)) > 0))
            print("Percentage of points that are not diffeomorphic: ", np.sum(np.linalg.det(J_phi) <= 0)/len(J_phi)*100, "percent")
            self.diffeo_mask=np.linalg.det(J_phi)<=0


        if  hasattr(self, 'training_delta'):
            self.training_delta = self.training_delta[:,:,np.newaxis]

            self.training_delta=  derivative_affine @ self.training_delta
            self.var_vel_transported=Jacobain_var @ self.training_delta**2

            self.training_delta= rot_gp @ self.training_delta
            self.training_delta=self.training_delta[:,:,0]
            self.var_vel_transported=self.var_vel_transported[:,:,0]


        if  hasattr(self, 'training_ori'):   
            quat_demo=quaternion.from_float_array(self.training_ori)
            quat_affine= quaternion.from_rotation_matrix(rot_affine)
            quat_gp = quaternion.from_rotation_matrix(rot_gp, nonorthogonal=True)
            quat_transport=quat_gp * (quat_affine * quat_demo)
            self.training_ori= quaternion.as_float_array(quat_transport)

    # def sample_transportation(self):
    #     delta_map_samples= self.gp_delta_map.samples(self.traj_rotated)
    #     training_traj_samples = self.traj_rotated + delta_map_samples 
    #     return training_traj_samples
