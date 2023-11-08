import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from policy_transportation import GaussianProcess as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
class KMP():
    def __init__(self, treshold_distance=5.0):
        self.treshold_distance=treshold_distance

    def find_matching_waypoints(self, source_distribution, training_traj):

        # Create KDTree for array2
        tree = cKDTree(source_distribution)

        # List to store pairs
        pairs = []
        matched_indices = set()  # To keep track of matched indices from array2
        self.mask_traj=np.zeros(len(training_traj), dtype=bool)
        # Iterate through each element in array1
        for i, element in enumerate(training_traj):
            # Query the KDTree for the nearest neighbor
            distance, idx = tree.query(element)
            
            # Check if the nearest neighbor has already been matched and if the distance is within threshold
            if distance <= self.treshold_distance and idx not in matched_indices:
                nearest_neighbor = source_distribution[idx]
                
                # Store the pair and mark the neighbor as matched
                pairs.append((element, nearest_neighbor))
                matched_indices.add(idx)
                self.mask_traj[i]=True

        self.mask_dist=np.array(list(matched_indices))

        return self.mask_traj, self.mask_dist

    def fit(self, source_distribution, target_distribution, training_traj):
        self.training_traj=training_traj
        self.time=np.linspace(0,1, self.training_traj.shape[0])
        diff=np.zeros_like(training_traj)
       
        diff[self.mask_traj]=target_distribution[self.mask_dist] - source_distribution[self.mask_dist]
        
        self.training_traj[self.mask_traj]= self.training_traj[self.mask_traj]+ diff[self.mask_traj]
        kernel=C(0.1) * RBF(length_scale=[0.1]) + WhiteKernel(3, [3, 1000])
        self.gp=GPR(kernel, n_targets=self.training_traj.shape[1])
        self.gp.fit(self.time.reshape(-1,1), self.training_traj)

    
    def predict(self, X, return_std=False):
        
        if return_std:
            self.mean, std= self.gp.predict(self.time.reshape(-1,1), return_std=True)
            return self.mean, std+np.sqrt(self.gp.kernel.get_params()['k2__noise_level'])
        else:
            self.mean, _ = self.gp.predict(self.time.reshape(-1,1), return_std=True)
            return self.mean
    
    def samples(self, X):
        samples, _=self.gp.predict(self.time.reshape(-1,1), return_std=True)
        samples=samples.reshape(1,-1,2)
        # samples=self.gp.samples(self.time.reshape(-1,1))
        return samples