"""
Authors: Giovanni Franzese 
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation import AffineTransform
from policy_transportation.models.torch.ensemble_neural_network import EnsembleNeuralNetwork as NeuralNetwork
import pickle
import numpy as np
import quaternion
class Ensemble_Neural_Transport():
    def __init__(self):
        super(Ensemble_Neural_Transport, self).__init__()


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


    def fit_transportation(self, num_epochs=20):
        if type(self.target_distribution) != type(self.source_distribution):
            raise TypeError("Both the distribution must be a numpy array.")
        elif not(isinstance(self.target_distribution, np.ndarray)) and not(isinstance(self.source_distribution, np.ndarray)):
            self.convert_distribution_to_array() #this needs to be a function of every sensor class

        self.affine_transform=AffineTransform()
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        source_distribution=self.affine_transform.predict(self.source_distribution)  
 
        delta_distribution = self.target_distribution - source_distribution

        self.delta_map=NeuralNetwork(source_distribution, delta_distribution)
        self.delta_map.fit(source_distribution, delta_distribution, num_epochs=num_epochs)  

    def apply_transportation(self):
              
        #Deform Trajactories 
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj)
        self.delta_map_mean, self.std= self.delta_map.predict(self.traj_rotated, return_std=True)
        self.training_traj = self.traj_rotated + self.delta_map_mean 

        #Deform Deltas and orientation
        if  hasattr(self, 'training_delta') or hasattr(self, 'training_ori'):
            pos=(np.array(self.traj_rotated))
            Jacobian, Jacobian_var=self.delta_map.derivative(pos, return_var=True)
            rot_gp= np.eye(Jacobian[0].shape[0]) + Jacobian
            derivative_affine= self.affine_transform.derivative(pos)


        if  hasattr(self, 'training_delta'):
            self.training_delta = self.training_delta[:,:,np.newaxis]

            self.training_delta=  derivative_affine @ self.training_delta
            self.var_vel_transported=Jacobian_var @ self.training_delta**2

            self.training_delta= rot_gp @ self.training_delta
            self.training_delta=self.training_delta[:,:,0]
            self.var_vel_transported=self.var_vel_transported[:,:,0]

            J_phi= rot_gp @ derivative_affine
            print("Is the map locally diffeomorphic?", np.all(np.linalg.det(J_phi)) > 0)
            print(np.linalg.det(J_phi))
            print("percerntagle of non-diffeomorphic points", np.sum(np.linalg.det(J_phi)<=0)/J_phi.shape[0])



    def sample_transportation(self):
        training_traj_samples= self.delta_map.samples(self.traj_rotated)
        return training_traj_samples
