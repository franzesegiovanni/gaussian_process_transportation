import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

class Laplacian_Editing():
    def __init__(self):
        pass

    def create_graph(self, training_traj):
        # Create a chain graph
        num_nodes = training_traj.shape[0]
        #check the distance between the first and the last point
        #if it is smaller than a threshold, create a cycle graph

        # The threshold should be the max distance between two consecutive points
        threshold_distance = 5* np.max(np.linalg.norm(training_traj[1:]-training_traj[:-1], axis=1))
        if np.linalg.norm(training_traj[0]-training_traj[-1])<threshold_distance:
            G = nx.cycle_graph(num_nodes)
            print("Cycle graph")
        else:
            G = nx.path_graph(num_nodes)    
            print("Path graph")
        # Compute the graph Laplacian matrix
        self.L = nx.laplacian_matrix(G).toarray()

        self.DELTA= self.L @ training_traj

        return self.L, self.DELTA

    def find_matching_waypoints(self, source_distribution, training_traj):
        # Threshold distance
        threshold_distance = 5.0

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
            if distance <= threshold_distance and idx not in matched_indices:
                nearest_neighbor = source_distribution[idx]
                
                # Store the pair and mark the neighbor as matched
                pairs.append((element, nearest_neighbor))
                matched_indices.add(idx)
                self.mask_traj[i]=True

        self.mask_dist=np.array(list(matched_indices))

        return self.mask_traj, self.mask_dist

    def fit(self, source_distribution, target_distribution, training_traj):
        self.training_traj=training_traj

        diff=np.zeros_like(training_traj)
        constraint= np.zeros_like(training_traj)

        L, DELTA= self.create_graph(training_traj)

        mask_traj, mask_dist= self.find_matching_waypoints(source_distribution, training_traj)
       
        diff[mask_traj]=target_distribution[mask_dist] - source_distribution[mask_dist]
        
        constraint[mask_traj]= training_traj[mask_traj]+ diff[mask_traj]

        P_hat=np.diag(1*mask_traj)

        A= np.vstack([L, P_hat])

        B= np.vstack([DELTA, constraint])

        P_s= np.linalg.pinv(A) @ B
        self.P_s= P_s[:len(training_traj),:]
    
    def predict(self, X, return_std=False):
        # assert that X adn self.training_traj have the same 
        # assert(np.allclose(X,self.training_traj)), " Laplacian editing can only predict the training trajectory"
        mean=self.P_s
        eps=1e-6
        if return_std:
            std = eps*np.ones_like(mean)
            return mean, std
        return mean
    
    def samples(self, X):
        # laplacian editing is deterministic, then we return the same sample
        predictions = [self.predict(X) for i in range(10)]
        predictions = np.array(predictions)  # Shape: (n_estimators, n_samples)
        return predictions