"""
Authors: Giovanni Franzese & Ravi Prakash, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation import  AffineTransform
import pickle
import numpy as np
import quaternion
import matplotlib.pyplot as plt
class AffineTransportation():
    def __init__(self):
        super(AffineTransportation, self).__init__()


    def fit_transportation(self, optimize=True, do_scale=False, do_rotation=True):
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.affine_transform.fit(self.source_distribution, self.target_distribution)
        self.scale= self.affine_transform.scale

    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.training_traj=self.affine_transform.predict(self.training_traj)

        #Deform Deltas and orientation
        if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
            pos=(np.array(self.training_traj))
            rot_affine= self.affine_transform.rotation_matrix
            derivative_affine= self.affine_transform.derivative(pos)


        if  hasattr(self, 'training_delta'):
            self.training_delta = self.training_delta[:,:,np.newaxis]

            self.training_delta=  derivative_affine @ self.training_delta
            self.var_vel_transported= np.zeros_like(self.training_delta)

            self.training_delta=self.training_delta[:,:,0]


        if  hasattr(self, 'training_ori'):   
            quat_demo=quaternion.from_float_array(self.training_ori)
            quat_affine= quaternion.from_rotation_matrix(rot_affine)
            quat_transport=(quat_affine * quat_demo)
            self.training_ori= quaternion.as_float_array(quat_transport)

    def sample_transportation(self):
        training_traj_samples = self.traj_rotated
        return training_traj_samples
