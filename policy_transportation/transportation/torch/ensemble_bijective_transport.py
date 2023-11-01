"""
Authors: Giovanni Franzese 
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation import AffineTransform
from policy_transportation.models.torch.ensemble_bijective_network import EnsembleBijectiveNetwork as BiJectiveNetwork
import pickle
import numpy as np
import quaternion
class Neural_Transport():
    def __init__(self):
        super(Neural_Transport, self).__init__()


    def save_distributions(self):
        # create a binary pickle file 
        f = open("distributions/source.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(self.source_distribution,f)
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


    def fit_transportation(self, num_epochs=20):
        if type(self.target_distribution) != type(self.source_distribution):
            raise TypeError("Both the distribution must be a numpy array.")
        elif not(isinstance(self.target_distribution, np.ndarray)) and not(isinstance(self.source_distribution, np.ndarray)):
            self.convert_distribution_to_array() #this needs to be a function of every sensor class

        self.affine_transform=AffineTransform()
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        # delta_distribution = self.target_distribution - source_distribution

        # self.gp_delta_map=NeuralNetwork(source_distribution, self.target_distribution)
        self.gp_delta_map=BiJectiveNetwork(source_distribution, self.target_distribution)

        self.gp_delta_map.fit(source_distribution, self.target_distribution , num_epochs=num_epochs)  

    def apply_transportation(self):
              
        #Deform Trajactories 
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        map_mean, self.std= self.gp_delta_map.predict(self.traj_rotated, return_std=True)
        # print(delta_map_mean)
        transported_traj =  map_mean 

        #Deform Deltas and orientation
        new_delta = np.ones_like(self.training_delta)
        for i in range(len(self.training_traj[:,0])):
            if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
                pos=(np.array(self.traj_rotated[i,:]).reshape(1,-1))
                # print(type(pos))
                Jacobian=self.gp_delta_map.derivative(pos)
                #print(Jacobian.shape)
                #print(Jacobian)
                Jacobian=Jacobian.reshape(self.training_delta.shape[1],pos.shape[1])
                #print(Jacobian)
                # Jacobian=np.zeros(pos.shape[1])
                rot_gp= Jacobian 
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

    def sample_transportation(self):
        training_traj_samples= self.gp_delta_map.samples(self.traj_rotated)
        return training_traj_samples

