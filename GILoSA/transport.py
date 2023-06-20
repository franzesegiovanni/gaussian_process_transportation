"""
Authors: Giovanni Franzese & Ravi Prakash, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from GILoSA import AffineTransform
from GILoSA.gaussian_process_torch import GaussianProcess 
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


    def fit_trasportation(self):
        if type(self.target_distribution) != type(self.source_distribution):
            raise TypeError("Both the distribution must be a numpy array.")
        elif not(isinstance(self.target_distribution, np.ndarray)) and not(isinstance(self.source_distribution, np.ndarray)):
            self.convert_distribution_to_array() #this needs to be a function of every sensor class

        self.affine_transform=AffineTransform()
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        delta_distribution = self.target_distribution - source_distribution

        if not(hasattr(self, 'kernel_transport')):
            self.kernel_transport=C(0.1) * RBF(length_scale=[0.1]) + WhiteKernel(0.0001) #this works for the surface
            print("Set kernel not set by the user")
        self.gp_delta_map=GaussianProcess(source_distribution, delta_distribution )
        self.gp_delta_map.fit()  

    def apply_trasportation(self):
              
        #Deform Trajactories 
        traj_rotated=self.affine_transform.predict(self.training_traj)
        delta_map_mean, _= self.gp_delta_map.predict(traj_rotated)
        print(delta_map_mean)
        transported_traj = traj_rotated + delta_map_mean 

        #Deform Deltas and orientation
        new_delta = np.ones_like(self.training_delta)
        for i in range(len(self.training_traj[:,0])):
            if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
                pos=(np.array(traj_rotated[i,:]).reshape(1,-1))
                #[Jacobian,_]=self.gp_delta_map.derivative(pos)
                #Jacobian=np.zeros(pos.shape[1])
                rot_gp= np.eye(2) #+ np.transpose(Jacobian[0]) 
                rot_affine= self.affine_transform.rotation_matrix
                if  hasattr(self, 'training_delta'):
                    new_delta[i]= rot_affine @ self.training_delta[i]
                    new_delta[i]= rot_gp @ new_delta[i]
                if  hasattr(self, 'training_ori'):
                    rot_gp_norm=rot_gp
                    rot_gp_norm=rot_gp_norm/np.linalg.det(rot_gp_norm)
                    quat_i=quaternion.from_float_array(self.training_ori[i,:])
                    rot_i=quaternion.as_rotation_matrix(quat_i)
                    rot_final=rot_gp_norm @ rot_affine @ rot_i
                    product_quat=quaternion.from_rotation_matrix(rot_final)
                    if quat_i.w*product_quat.w  + quat_i.x * product_quat.x+ quat_i.y* product_quat.y + quat_i.z * product_quat.z < 0:
                        product_quat = - product_quat
                    self.training_ori[i,:]=np.array([product_quat.w, product_quat.x, product_quat.y, product_quat.z])

        #Update the trajectory and the delta     
        self.training_traj=transported_traj
        if  hasattr(self, 'training_delta'):
            self.training_delta=new_delta
