"""
Authors: Giovanni Franzese 
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation import AffineTransform
from policy_transportation.models.torch.stocastic_variational_gaussian_process_derivatives import StocasticVariationalGaussianProcess 
# from policy_transportation.models.torch.stocastic_variational_gaussian_process import StocasticVariationalGaussianProcess
import pickle
import numpy as np
import quaternion
class SVGPTransport():
    def __init__(self):
        super(SVGPTransport, self).__init__()


    def fit_transportation(self, num_epochs=20, num_inducing=100):
        if type(self.target_distribution) != type(self.source_distribution):
            raise TypeError("Both the distribution must be a numpy array.")
        elif not(isinstance(self.target_distribution, np.ndarray)) and not(isinstance(self.source_distribution, np.ndarray)):
            self.convert_distribution_to_array() #this needs to be a function of every sensor class

        self.affine_transform=AffineTransform()
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        delta_distribution = self.target_distribution - source_distribution

        self.gp_delta_map=StocasticVariationalGaussianProcess(source_distribution, delta_distribution, num_inducing=num_inducing)
        self.gp_delta_map.fit(num_epochs=num_epochs)  

    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.delta_map_mean, self.std= self.gp_delta_map.predict(self.traj_rotated, return_std=True)
        # convert to numpyu array
        self.delta_map_mean=self.delta_map_mean
        self.std=self.std
        self.training_traj = self.traj_rotated + self.delta_map_mean 

        #Deform Deltas and orientation
        if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
            pos=(np.array(self.traj_rotated))
            Jacobian, Jacobian_var=self.gp_delta_map.derivative(pos, return_var=True)
            Jacobian=self.gp_delta_map.derivative(pos)

            rot_gp= np.eye(Jacobian[0].shape[0]) + Jacobian
            rot_affine= self.affine_transform.rotation_matrix
            derivative_affine= self.affine_transform.derivative(pos)


        if  hasattr(self, 'training_delta'):
            self.training_delta = self.training_delta[:,:,np.newaxis]

            self.training_delta=  derivative_affine @ self.training_delta
            self.var_vel_transported=Jacobian_var @ self.training_delta**2

            self.training_delta= rot_gp @ self.training_delta
            self.training_delta=self.training_delta[:,:,0]
            self.var_vel_transported=self.var_vel_transported[:,:,0]

            J_phi= rot_gp @ derivative_affine

            print("Is the map locally diffeomorphic?", np.all(np.linalg.det(J_phi)> 0)) 
            print("percerntagle of non-diffeomorphic points", np.sum(np.linalg.det(J_phi)<=0)/len(J_phi))
            self.diffeo_mask=np.linalg.det(J_phi)<=0



        if  hasattr(self, 'training_ori'):   
            quat_demo=quaternion.from_float_array(self.training_ori)
            quat_affine= quaternion.from_rotation_matrix(rot_affine)
            quat_gp = quaternion.from_rotation_matrix(rot_gp, nonorthogonal=True)
            quat_transport=quat_gp * (quat_affine * quat_demo)
            self.training_ori= quaternion.as_float_array(quat_transport)

