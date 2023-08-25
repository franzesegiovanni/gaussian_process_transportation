"""
Authors: Giovanni Franzese & Ravi Prakash, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from GILoSA import GaussianProcess, AffineTransform
import pickle
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import quaternion
class Transport():
    def __init__(self):
        super(Transport, self).__init__()


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


    def fit_transportation(self, optimize=True):
        self.affine_transform=AffineTransform()
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        delta_distribution = self.target_distribution - source_distribution

        if not(hasattr(self, 'kernel_transport')):
            self.kernel_transport=C(0.1) * RBF(length_scale=[0.1]) + WhiteKernel(0.0001) #this works for the surface
            print("Kernel not set by the user")
            print("Kernel:", self.kernel_transport)
        if optimize==True:    
            self.gp_delta_map=GaussianProcess(kernel=self.kernel_transport, n_restarts_optimizer=5)
            self.kernel_transport=self.gp_delta_map.kernel
        else:
            self.gp_delta_map=GaussianProcess(kernel=self.kernel_transport, optimizer=None)    
        self.gp_delta_map.fit(source_distribution, delta_distribution)  

    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        delta_map_mean, self.std= self.gp_delta_map.predict(self.traj_rotated, return_std=True)
        self.training_traj = self.traj_rotated + delta_map_mean 

        #Deform Deltas and orientation
        for i in range(len(self.training_traj[:,0])):
            if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
                pos=(np.array(self.traj_rotated[i,:]).reshape(1,-1))
                [Jacobian,_]=self.gp_delta_map.derivative(pos)
                rot_gp= np.eye(Jacobian[0].shape[0]) + np.transpose(Jacobian[0]) 
                rot_affine= self.affine_transform.rotation_matrix
                derivative_affine= self.affine_transform.derivative(pos)
                if  hasattr(self, 'training_delta'):
                    self.training_delta[i]= derivative_affine @ self.training_delta[i]
                    self.training_delta[i]= rot_gp @ self.training_delta[i]
                if  hasattr(self, 'training_ori'):
                    rot_gp_norm=rot_gp/np.linalg.det(rot_gp)
                    quat_i=quaternion.from_float_array(self.training_ori[i,:])
                    rot_i=quaternion.as_rotation_matrix(quat_i)
                    rot_final=rot_gp_norm @ rot_affine @ rot_i
                    product_quat=quaternion.from_rotation_matrix(rot_final)
                    if quat_i.w*product_quat.w  + quat_i.x * product_quat.x+ quat_i.y* product_quat.y + quat_i.z * product_quat.z < 0:
                        product_quat = - product_quat
                    self.training_ori[i,:]=np.array([product_quat.w, product_quat.x, product_quat.y, product_quat.z])
                if hasattr(self, 'training_stiff_ori'):
                    rot_stiff=rot_gp_norm @ rot_affine
                    quat_stiff=quaternion.from_rotation_matrix(rot_stiff)
                    self.training_stiff_ori[i,:]=np.array([quat_stiff.w, quat_stiff.x, quat_stiff.y, quat_stiff.z])
    
    def check_invertibility(self):
        y= self.traj_rotated - self.training_traj
        x= self.training_traj
        gp_delta_map_inv=GaussianProcess(kernel=self.kernel_transport, n_restarts_optimizer=5)
        gp_delta_map_inv.fit(x, y)
        delta_map_mean, self.std= gp_delta_map_inv.predict(x, return_std=True)
        traj_rotated_inv=delta_map_mean+ self.training_traj
        score=gp_delta_map_inv.gp.score(x, y)
        print("Score", gp_delta_map_inv.gp.score(x, y))
        # print("Error in the transportation map:", np.linalg.norm(self.traj_rotated-traj_rotated_inv))
        print( 'The transportation map is inveritible')
        print(score>0.99)
        # print(np.all(np.abs(traj_rotated_inv-  self.traj_rotated) < tollerance))

    def fit_transportation_linear(self):
        self.affine_transform=AffineTransform()
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

    def apply_transportation_linear(self):
            #Deform Trajactories 
        self.training_traj=self.affine_transform.predict(self.training_traj)

        #Deform Deltas and orientation
        for i in range(len(self.training_traj[:,0])):
            if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
                pos=(np.array(self.training_traj[i,:]).reshape(1,-1))
                rot_affine= self.affine_transform.rotation_matrix
                derivative_affine= self.affine_transform.derivative(pos)
                if  hasattr(self, 'training_delta'):
                    self.training_delta[i]= derivative_affine @ self.training_delta[i]
                if  hasattr(self, 'training_ori'):
                    quat_i=quaternion.from_float_array(self.training_ori[i,:])
                    rot_i=quaternion.as_rotation_matrix(quat_i)
                    rot_final=rot_affine @ rot_i
                    product_quat=quaternion.from_rotation_matrix(rot_final)
                    if quat_i.w*product_quat.w  + quat_i.x * product_quat.x+ quat_i.y* product_quat.y + quat_i.z * product_quat.z < 0:
                        product_quat = - product_quat
                    self.training_ori[i,:]=np.array([product_quat.w, product_quat.x, product_quat.y, product_quat.z])
                if hasattr(self, 'training_stiff_ori'):
                    rot_stiff= rot_affine
                    quat_stiff=quaternion.from_rotation_matrix(rot_stiff)
                    self.training_stiff_ori[i,:]=np.array([quat_stiff.w, quat_stiff.x, quat_stiff.y, quat_stiff.z])

def is_rotation_matrix(matrix):
    # Check if the matrix is orthogonal
    is_orthogonal = np.allclose(np.eye(matrix.shape[0]), matrix @ matrix.T)
    if not is_orthogonal:
        return False

    # Check if the determinant of the matrix is 1
    det = np.linalg.det(matrix)
    if not np.isclose(det, 1.0):
        return False

    return True