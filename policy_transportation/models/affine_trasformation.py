"""
Authors: Giovanni Franzese, Ravi Prakash March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import numpy as np
class AffineTransform():
    def __init__(self, do_scale=False, do_rotation=True):
        self.do_scale = do_scale
        self.do_rotation = do_rotation
        self.scale = 1
        pass

    def fit(self, source_points, target_points):
        # Matched points should have the same length
        assert len(source_points) == len(target_points)
            
        # Compute centroids
        self.S_centroid = np.mean(source_points, axis=0)
        self.T_centroid = np.mean(target_points, axis=0)
        self.source_points_centered=source_points-self.S_centroid
        self.target_points_centered=target_points-self.T_centroid
        H = np.dot(np.transpose(self.source_points_centered),  self.target_points_centered)
        rank_H = np.linalg.matrix_rank(H)

        if not self.do_rotation:
            self.rotation_matrix= np.eye(source_points.shape[1])
            print("Rotation matrix is not computed and set to identity...")
        elif rank_H < source_points.shape[1]:
            self.rotation_matrix= np.eye(source_points.shape[1])
            print("Rotation matrix cannot be uniquely determined. Set to identity...")
        else:   
	        # Perform SVD
            U, S, Vt = np.linalg.svd(H)
            V=Vt.T
            self.rotation_matrix = V @ U.T 
            #Check for reflactions https://nghiaho.com/?page_id=671
            if np.linalg.det(self.rotation_matrix)<0:
                V[:,-1]*= -1
                self.rotation_matrix= V @ U.T

        if self.do_scale:
            source_rotated=np.transpose(self.rotation_matrix @ np.transpose((self.source_points_centered)))
            self.scale = np.sum(source_rotated * self.target_points_centered) / np.sum(source_rotated**2)
        print ("Rotation Matrix of the Affine Matrix:")
        print(self.rotation_matrix)
        print ("Scaling factor:", self.scale)
        #Compute translation
        self.translation=self.T_centroid-self.S_centroid
        
    def predict(self, x):
        transported_x= self.scale*np.transpose(self.rotation_matrix @ np.transpose((x-self.S_centroid)))+ self.T_centroid
        return transported_x
        
    def derivative(self,x):
        affine_derivative=np.repeat(self.rotation_matrix[np.newaxis, :, :], x.shape[0], axis=0)
        return affine_derivative
    
