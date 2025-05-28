"""
Authors: Giovanni Franzes, June 2024
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation import AffineTransform
import numpy as np
import quaternion
class PolicyTransportation():  
    def __init__(self, method=None, is_residual=True):
        super(PolicyTransportation, self).__init__()
        self.nonlinear_transform=method
        self.is_residual=is_residual
    def set_method(self, method, is_residual=True):
        """
        Set the method for the nonlinear transformation. You can use any method that implements the fit and predict methods, such as Iterative_Locally_Weighted_Translations or GaussianProcess.
        Parameters:
            method (object): The method to be used for the nonlinear transformation. It should implement fit and predict methods.
            is_residual (bool): If True, Phi(x)= Psi(Gamma(x)) + Gamma(x), where Gamma is the affine transformation and Psi is the nonlinear transformation. If False, Phi(x)=Psi(Gamma(x)). Default is True.
        """

        self.nonlinear_transform=method
        self.is_residual=is_residual

    def fit(self, source_distribution, target_distribution, do_scale=False, do_rotation=True):
        if self.nonlinear_transform is None:
            raise ValueError("Nonlinear transform method is not set. Please set it using set_method() before fitting.")
        if source_distribution.shape[0] != target_distribution.shape[0]:
            raise ValueError("Source and target distributions must have the same number of points.")
        if source_distribution.shape[1] != target_distribution.shape[1]:
            raise ValueError("Source and target distributions must have the same dimensionality.")
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.affine_transform.fit(source_distribution, target_distribution)

        source_distribution_rotated=self.affine_transform.predict(source_distribution)  
        if self.is_residual==True:
            self.delta_distribution = target_distribution - source_distribution_rotated
            self.nonlinear_transform.fit(source_distribution_rotated, self.delta_distribution)  
        
        else:
            self.nonlinear_transform.fit(source_distribution_rotated, target_distribution)  

    def transport(self, pos, return_std=True):
        """
        Transport positions using the learned transformation.
        
        Parameters:
            pos (numpy.ndarray): Input positions, shape (n, 2) or (n, 3) where n is the number of points.
            return_std (bool): Whether to return the standard deviation.
            
        Returns:
            numpy.ndarray: Transported positions, same shape as input.
            numpy.ndarray: Standard deviation of the transport if return_std=True, same shape as input.
        """
        pos_rotated = self.affine_transform.predict(pos)
        if return_std:
            delta_map_mean, nonlinear_map_std = self.nonlinear_transform.predict(pos_rotated, return_std=return_std)
        else:
            delta_map_mean = self.nonlinear_transform.predict(pos_rotated, return_std=return_std)

        if self.is_residual:
            pos_transported = pos_rotated + delta_map_mean
        else:
            pos_transported = delta_map_mean 
        if return_std:
            return pos_transported, nonlinear_map_std
        else:
            return pos_transported
    
    def compute_jacobian(self, pos, return_var=True):
        pos_rotated=self.affine_transform.predict(pos)
        J_gamma= self.affine_transform.derivative(pos)
        if return_var==True:
            J_psi, J_psi_var=self.nonlinear_transform.derivative(pos_rotated, return_var=return_var)
        else:
            J_psi= self.nonlinear_transform.derivative(pos_rotated, return_var=return_var)
        
        if self.is_residual==True:
            J_phi= J_gamma + J_psi @ J_gamma
        else:
            J_phi= J_psi @ J_gamma
        
        if return_var:
            return J_phi, J_psi_var
        else:
            return J_phi

    def transport_velocity(self, pos, vel, return_var=True):
        """
        Transport velocities using the learned transformation.
        Parameters:
            pos (numpy.ndarray): Input positions, shape (n, 2) or (n, 3) where n is the number of points.
            vel (numpy.ndarray): Input velocities, shape (n, 2) or (n, 3) where n is the number of points.
            return_var (bool): Whether to return the variance of the transported velocities.
        Returns:
            numpy.ndarray: Transported velocities, same shape as input.
            numpy.ndarray: Variance of the transported velocities if return_var=True, same shape as input.
        """
        if return_var:
            J_phi, J_psi_var = self.compute_jacobian(pos, return_var=return_var)
        else:
            J_phi = self.compute_jacobian(pos, return_var=return_var)

    
        vel = vel[:,:,np.newaxis]
        vel_transported= J_phi @ vel
        vel_transported=vel_transported[:,:,0]

        if return_var==True:
            J_gamma= self.affine_transform.derivative(pos)
            vel_rotated=  J_gamma @ vel
            var_vel_transported=J_psi_var @ vel_rotated**2
            var_vel_transported=var_vel_transported[:,:,0]
            return vel_transported, var_vel_transported
        else:
            return vel_transported
        
    def transport_orientation(self, pos, ori):
        """
        Transport orientations using the learned transformation.
        Parameters:
            pos (numpy.ndarray): Input positions, shape (n, 2) or (n, 3) where n is the number of points.
            ori (numpy.ndarray): Input orientations, shape (n, 4) where n is the number of points and each orientation is represented as a quaternion. The quaternion should be in the alphabetic order [w, x, y, z].
        """
        J_phi= self.compute_jacobian(pos, return_var=False)

        if J_phi[0].shape[0]==3:
            quat=quaternion.from_float_array(ori)
            quat_J_phi = quaternion.from_rotation_matrix(J_phi, nonorthogonal=True)
            quat_transport=quat_J_phi * quat
            ori_transported= quaternion.as_float_array(quat_transport)

            return ori_transported

        else:
            print("The Jacobain of the map as shape ", self.J_phi[0].shape, " but it should be (3x3)")
            print("Robot orientation is not transported")

    def sample_transportation(self, pos):
        pos_rotated=self.affine_transform.predict(pos)
        nonlinear_samples= self.nonlinear_transform.samples(pos_rotated)
        if self.is_residual==True:
            training_traj_samples = pos_rotated + nonlinear_samples 
        else:
            training_traj_samples = nonlinear_samples
        return training_traj_samples
    
    def is_diffeomorphic_on(self, pos):
        pos_rotated=self.affine_transform.predict(pos)
        J_gamma= self.affine_transform.derivative(pos)
        J_psi= self.nonlinear_transform.derivative(pos_rotated, return_var=False)
        if self.is_residual==True:
            J_phi= J_gamma + J_psi @ J_gamma
        else:
            J_phi= J_psi @ J_gamma
        print("Is the map diffeomorphic?", np.all((np.linalg.det(J_phi)) > 0))
        print("Percentage of points that are not diffeomorphic: ", np.sum(np.linalg.det(J_phi) <= 0)/len(J_phi)*100, "percent")
        return np.linalg.det(J_phi) > 0


