"""
Authors: Ravi Prakash & Giovanni Franzese, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import numpy as np
class AffineTransform():
    def __init__(self):
        pass

    def fit(self, source_points, target_points):
        # Matched points should have the same length
        assert len(source_points) == len(target_points)

        # Compute centroids
        self.S_centroid = np.mean(source_points, axis=0)
        self.T_centroid = np.mean(target_points, axis=0)
        self.source_points_centered=source_points-self.S_centroid
        self.target_points_centered=target_points-self.T_centroid

        #  Compute covariance matrix
        H = np.dot(np.transpose(self.source_points_centered),  self.target_points_centered)
	    # Perform SVD
        U, S, Vt = np.linalg.svd(H)
        V=Vt.T
        self.rotation_matrix = V @ U.T 
        #Check for reflactions https://nghiaho.com/?page_id=671
        if np.linalg.det(self.rotation_matrix)<0:
            V[:,-1]*= -1
            self.rotation_matrix= V @ U.T
        print ("Rotation Matrix:", self.rotation_matrix)
        #Compute translation
        self.translation=self.T_centroid-self.S_centroid
        
    def predict(self, x):
        transported_x= np.transpose(self.rotation_matrix @ np.transpose((x-self.S_centroid)))+ self.T_centroid
        return transported_x
        
    def derivative(self,x):  
        for i in range(num_points):
            transport_derivative[i] = self.rotation_matrix  
        return transport_derivative[i]
