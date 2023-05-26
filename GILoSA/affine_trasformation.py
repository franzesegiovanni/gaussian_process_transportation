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

        # Create affine transformation matrix
        self.rotation_matrix=np.linalg(source_points_centered) @ target_points_centered
        self.rotation_matrix=self.rotation_matrix/np.linalg.det(self.rotation_matrix)
        self.translation=target_points_centered-source_points_centered
    def predict(self, x):
        transported_x=(self.T_centroid-self.S_centroid)+(x-self.S_centroid)*self.rotation_matrix
        return transported_x
        
    def derivative(self,x):  
        for i in range(num_points):
            transport_derivative[i] = self.rotation_matrix  
        return transport_derivative[i]