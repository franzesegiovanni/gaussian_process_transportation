import numpy as np
class AffineTransform():
    def __init__(self):
        pass

    def fit(self, source_points, target_points):
        # Matched points should have the same length
        assert len(source_points) == len(target_points)

        # Compute centroids
        self.S_centroid = np.mean(source_points, axis=0)
        print("Mean Source", self.S_centroid)
        self.T_centroid = np.mean(target_points, axis=0)
        print("Mean Target", self.T_centroid)
        self.source_points_centered=source_points-self.S_centroid
        self.target_points_centered=target_points-self.T_centroid

        #  Compute covariance matrix
        H = np.dot(np.transpose(self.source_points_centered),  self.target_points_centered)
        print("covariance matrix")
        print(H)
	    # Perform SVD
        U, S, Vt = np.linalg.svd(H)

        #Use Kabsh algorithm https://en.wikipedia.org/wiki/Kabsch_algorithm
        print(S.shape[0])
        S_prime=np.eye(S.shape[0])
        S_prime[-1,-1]=np.sign(np.linalg.det(Vt.T*U.T))*1
        print("S prime:", S_prime)
        # Compute Rotation Matrix
        self.rotation_matrix= Vt.T @ S_prime @ U.T
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
