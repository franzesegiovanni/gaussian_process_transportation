import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
import numpy as np
from scipy.optimize import linear_sum_assignment
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


       # ceate cdist matrix
        distance_matrix = np.linalg.norm(training_traj[:, None] - source_distribution, axis=2)


        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        return row_ind, col_ind

    
    def fit(self, source_distribution, target_distribution, training_traj):
        self.training_traj=training_traj

        diff=np.zeros_like(training_traj)
        constraint= np.zeros_like(training_traj)

        L, DELTA= self.create_graph(training_traj)

        mask_traj, mask_dist= self.find_matching_waypoints(source_distribution, training_traj)
       
        diff[mask_traj]=target_distribution[mask_dist] - source_distribution[mask_dist]
        
        constraint[mask_traj]= training_traj[mask_traj]+ diff[mask_traj]

        # make a vector that has 1 in the index of mask_traj

        vect= np.zeros(len(training_traj))

        vect[mask_traj]=1
        P_hat=np.diag(vect)


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