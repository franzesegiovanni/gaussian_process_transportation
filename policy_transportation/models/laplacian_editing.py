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
        # Compute the graph Laplacian matrix that is the discrete analog of the Laplace-Beltrami operator. It can be computed as the difference between the degree matrix and the adjacency matrix.The degree matrix of an undirected graph is a diagonal matrix which contains information about the degree of each vertex—that is, the number of edges attached to each vertex. The adjacency matrix of an undirected graph is a square matrix with dimensions equal to the number of vertices in the graph. The elements of the matrix indicate whether pairs of vertices are adjacent or not in the graph.
        self.L = nx.laplacian_matrix(G).toarray()
        self.L = self.L 
        # Rather than working in absolute Cartesian coordinates, the discrete Laplace-Beltrami operator specifies the loca path properties, called Laplacian coordinates Delta that can be calculated as the product of the graph Laplacian and the training trajectory
        self.DELTA= self.L @ training_traj

        return self.L, self.DELTA
    
    def find_matching_waypoints(self, source_distribution, training_traj):


       # ceate cdist matrix
        distance_matrix = np.linalg.norm(training_traj[:, None] - source_distribution, axis=2)


        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        # The Hungarian algorithm is a combinatorial optimization algorithm that solves the assignment problem in polynomial time. The algorithm has many applications in combinatorial optimization, for example in problems of matching supply and demand in transportation networks, or in finding the minimum cost assignment in job scheduling. The algorithm is also known as the Kuhn-Munkres algorithm.

        mask= np.linalg.norm(training_traj[row_ind] - source_distribution[col_ind],axis=1) < 5
        row_ind=row_ind[mask]
        col_ind=col_ind[mask]
        return row_ind, col_ind

    
    def fit(self, source_distribution, target_distribution, training_traj):
        self.training_traj=training_traj

        diff=np.zeros_like(training_traj)
        constraint= np.zeros_like(training_traj)

        L, DELTA= self.create_graph(training_traj)

        mask_traj, mask_source= self.find_matching_waypoints(source_distribution, training_traj)
        self.mask_traj=mask_traj
        self.mask_source=mask_source
        diff[mask_traj]=target_distribution[mask_source] - source_distribution[mask_source]
        
        constraint[mask_traj]= training_traj[mask_traj]+ diff[mask_traj]

        # make a vector that has 1 in the index of mask_traj

        vect= np.zeros(len(training_traj))
        vect[mask_traj]=1
        P_hat=np.diag(vect)
        # P_hat is a diagonal matrix that has 1 in the index of mask_traj and 0 otherwise

        # We are now solving the following optimization problem
        # min ||L P_hat - DELTA||^2 + ||P_hat - constraint||^2 and this can be written as
        # min ||A P_s - B||^2 where A= [L; P_hat] and B= [DELTA; constraint] and P_s is the solution of the optimization problem. Since it is a linear system with more constratint than variables, the solution is given by the pseudo inverse of A @ B
        weight_delta= 1
        weight_constraint=1
        A= np.vstack([L*weight_delta, P_hat*weight_delta])

        B= np.vstack([DELTA*weight_constraint, constraint*weight_constraint])
        

        self.P_s, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

        # The solution is the new trajectory where the assigned nodes to the source are moved by the quantity specified by the difference between the target and the source distribution

        # Check out much the solution of the syste respected the orginal contraint in B
        # print("Residuals: ", A @ self.P_s - B)
        # print("Residuals on Delta", 100*(L @ self.P_s - DELTA)/DELTA)
        # print("Residuals on constraint", 100*(P_hat @ self.P_s - constraint)/constraint)
        self.accuracy = np.sqrt(np.mean((P_hat @ self.P_s - constraint)**2))

    
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