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



    def Policy_Transport(self):
        if type(self.target_distribution) != type(self.source_distribution):
            raise TypeError("Both the distribution must be a numpy array.")
        elif not(isinstance(self.target_distribution, np.ndarray)) and not(isinstance(self.source_distribution, np.ndarray)):
            self.convert_distribution_to_array() #this needs to be a function of every sensor class

        affine_transform=AffineTransform()
        affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=affine_transform.predict(self.source_distribution)  
 
        delta_distribution = self.target_distribution - source_distribution

        #kernel = C(0.1) * RBF(length_scale=np.ones(source_distribution.shape[1])) + WhiteKernel(0.0001, [0.0001,0.0001])
        #kernel = C(0.1) * RBF() + WhiteKernel(0.0001, [0.0001,0.0001])
        #kernel = C() * RBF() + WhiteKernel()
        # kernel = C(0.1) * RBF(length_scale=np.ones(source_distribution.shape[1]), length_scale_bounds=[0.1, 1]) + WhiteKernel()
        # kernel = C(0.1,[0.1,0.1]) * RBF(length_scale=[0.5], length_scale_bounds=[0.3,1.0]) + WhiteKernel(0.0001, [0.0001,0.0001]) # test based on sim
        #kernel = C(0.1,[0.1,0.1]) * RBF(length_scale=[0.5]) + WhiteKernel(0.0001, [0.0001,0.0001]) #working for tags
        kernel = C(0.1) * RBF(length_scale=[0.1]) + WhiteKernel(0.0001) #this works for the surface
        self.gp_delta_map=GaussianProcess(kernel=kernel, n_restarts_optimizer=5)
        print(kernel)
        print(source_distribution.shape)
        self.gp_delta_map.fit(source_distribution, delta_distribution)        
        
        #Deform Trajactories 
        traj_rotated=affine_transform.predict(self.training_traj)
        delta_map_mean, _= self.gp_delta_map.predict(traj_rotated)
        transported_traj = traj_rotated + delta_map_mean 

        #Deform Deltas and orientation
        new_delta = np.ones_like(self.training_delta)
        for i in range(len(self.training_traj[:,0])):
            if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
                pos=(np.array(traj_rotated[i,:]).reshape(1,-1))
                [Jacobian,_]=self.gp_delta_map.derivative(pos)
                rot_gp= np.eye(Jacobian[0].shape[0]) + np.transpose(Jacobian[0]) 
                rot_affine= affine_transform.rotation_matrix
                if  hasattr(self, 'training_delta'):
                    new_delta[i]= rot_affine @ self.training_delta[i]
                    new_delta[i]= rot_gp @ new_delta[i]
                if  hasattr(self, 'training_ori'):
                    rot_gp_norm=rot_gp
                    rot_gp_norm=rot_gp_norm/np.linalg.det(rot_gp_norm)
                    # U, S, Vt = np.linalg.svd(rot_gp)
                    # V=Vt.T
                    # rot_gp_norm = V @ U.T 
                    #Check for reflactions https://nghiaho.com/?page_id=671
                    # if np.linalg.det(rot_gp_norm)<0:
                    #     V[:,-1]*= -1
                    #     rot_gp_norm= V @ U.T
                    # Assert that this is a rotation matrix
                    # assert(is_rotation_matrix(rot_gp_norm))    
                    #print("Rotation gp:" , rot_gp_norm)
                    #quat_deformation=quaternion.from_rotation_matrix(rot_gp_norm)
                    #quat_deformation=quaternion.from_rotation_matrix(rot_affine)
                    # quat_deformation=quaternion.from_rotation_matrix(rot_gp_norm @ rot_affine)
                    # quat_deformation=np.sign(quat_deformation.w)*quat_deformation
                    quat_i=quaternion.from_float_array(self.training_ori[i,:])
                    rot_i=quaternion.as_rotation_matrix(quat_i)
                    rot_final=rot_gp_norm @ rot_affine @ rot_i
                    #dot_product = quaternion.dot_product(quat_i, quat_deformation)
                    # print(dot_product)
                    # if quat_i.w*quat_deformation.w  + quat_i.x * quat_deformation.x+ quat_i.y* quat_deformation.y + quat_i.z * quat_deformation.z < 0:
                    #     quat_deformation = -quat_deformation
                    #     print("Invertion")
                    # print("Rot Affine")
                    # print(rot_affine)
                    # print("Euler_angle")
                    # print(quat_deformation)    
                    # product_quat=quat_deformation*quat_i
                    product_quat=quaternion.from_rotation_matrix(rot_final)
                    if quat_i.w*product_quat.w  + quat_i.x * product_quat.x+ quat_i.y* product_quat.y + quat_i.z * product_quat.z < 0:
                        product_quat = - product_quat
                    self.training_ori[i,:]=np.array([product_quat.w, product_quat.x, product_quat.y, product_quat.z])

        #Update the trajectory and the delta     
        self.training_traj=transported_traj
        if  hasattr(self, 'training_delta'):
            self.training_delta=new_delta
