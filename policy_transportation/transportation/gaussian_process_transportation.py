"""
Authors: Giovanni Franzese & Ravi Prakash, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import optuna
from policy_transportation import GaussianProcess, AffineTransform
import pickle
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import quaternion
import matplotlib.pyplot as plt
class GaussianProcessTransportation():
    def __init__(self, kernel_transport=C(0.1) * RBF(length_scale=[0.1]) + WhiteKernel(0.0001)):
        super(GaussianProcessTransportation, self).__init__()
        self.kernel_transport=kernel_transport

    def save_distributions(self):
        # create a binary pickle file 
        f = open("distributions/source.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(self.source_distribution,f)
        # close file
        f.close()

    # create a binary pickle file 
        f = open("distributions/target.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(self.target_distribution,f)
        # close file
        f.close()

    def load_distributions(self):
        try:
            with open("distributions/source.pkl","rb") as source:
                self.source_distribution = pickle.load(source)
        except:
            print("No source distribution saved")

        try:
            with open("distributions/target.pkl","rb") as target:
                self.target_distribution = pickle.load(target)
        except:
            print("No target distribution saved")    


    def fit_transportation(self, optimize=True, do_scale=False, do_rotation=True):
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        self.delta_distribution = self.target_distribution - source_distribution

            
        print("Kernel:", self.kernel_transport)
        if optimize==True:    
            self.gp_delta_map=GaussianProcess(kernel=self.kernel_transport, n_restarts_optimizer=5)
        else:
            self.gp_delta_map=GaussianProcess(kernel=self.kernel_transport, optimizer=None)    
        self.gp_delta_map.fit(source_distribution, self.delta_distribution)  
        self.kernel_transport=self.gp_delta_map.kernel


    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.delta_map_mean, self.std= self.gp_delta_map.predict(self.traj_rotated, return_std=True)
        self.training_traj = self.traj_rotated + self.delta_map_mean 

        #Deform Deltas and orientation
        if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
            pos=(np.array(self.traj_rotated))
            J_psi, J_psi_var=self.gp_delta_map.derivative(pos, return_var=True)

            J_gamma= self.affine_transform.derivative(pos)

            J_phi= J_gamma + J_psi @ J_gamma

            print("Is the map locally diffeomorphic?", np.all(np.linalg.det(J_phi) > 0))

        if  hasattr(self, 'training_delta'):
            self.training_delta = self.training_delta[:,:,np.newaxis]

            training_delta_rotated=  J_gamma @ self.training_delta
            self.var_vel_transported=J_psi_var @ training_delta_rotated**2

            self.training_delta= J_phi @ self.training_delta
            self.training_delta=self.training_delta[:,:,0]
            self.var_vel_transported=self.var_vel_transported[:,:,0]


        if  hasattr(self, 'training_ori'):   
            if J_phi[0].shape[0]==3:
        
                quat_demo=quaternion.from_float_array(self.training_ori)
                quat_gp = quaternion.from_rotation_matrix(J_phi, nonorthogonal=True)
                quat_transport=quat_gp * quat_demo
                self.training_ori= quaternion.as_float_array(quat_transport)
            else:
                print("The Jacobain of the map as shape ", J_phi[0].shape, " but it should be (3x3)")
                print("Robot orientation is not transported")


    def sample_transportation(self):
        delta_map_samples= self.gp_delta_map.samples(self.traj_rotated)
        training_traj_samples = self.traj_rotated + delta_map_samples 
        return training_traj_samples


