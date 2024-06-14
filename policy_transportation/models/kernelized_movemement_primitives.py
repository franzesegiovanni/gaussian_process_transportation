import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from policy_transportation import GaussianProcess as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
class KMP():
    def __init__(self):
        pass

    def find_matching_waypoints(self, source_distribution, training_traj):


       # ceate cdist matrix
        distance_matrix = np.linalg.norm(training_traj[:, None] - source_distribution, axis=2)


        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        return row_ind, col_ind

    def fit(self, source_distribution, target_distribution, training_traj, kernel=C(0.1, constant_value_bounds=[0.1,  10]) * RBF(length_scale=[0.1], length_scale_bounds=[0.3, 1]) + WhiteKernel(0.00001)):
        self.training_traj=training_traj
        self.time=np.linspace(0,1, self.training_traj.shape[0])


        self.gp=GPR(kernel, n_targets=self.training_traj.shape[1])
        self.gp.fit(self.time.reshape(-1,1), self.training_traj)
        self.noise_var_ = self.gp.alpha + self.gp.kernel.get_params()['k2__noise_level']

        # diff=np.zeros_like(training_traj)
       
        k_star= self.gp.kernel(self.time.reshape(-1,1), self.time[self.mask_traj].reshape(-1,1))
        K_star_star = self.gp.kernel(self.time[self.mask_traj].reshape(-1,1), self.time[self.mask_traj].reshape(-1,1))
        K_star_star_noise = K_star_star + np.eye(K_star_star.shape[0])*self.noise_var_
        self.noise_var_ = self.gp.alpha + self.gp.kernel.get_params()['k2__noise_level']
        self.training_traj = self.training_traj + k_star @ np.linalg.inv(K_star_star_noise) @ (target_distribution[self.mask_dist] - source_distribution[self.mask_dist])
        print("Fitting the transportation function")
        self.gp.fit(self.time.reshape(-1,1), self.training_traj)

        self.transportation_variance = self.gp.kernel(self.time.reshape(-1,1), self.time.reshape(-1,1)) - k_star @ np.linalg.inv(K_star_star_noise) @ k_star.T

        self.transportation_variance = np.repeat(self.transportation_variance[:,:,np.newaxis], self.training_traj.shape[1], axis=2)


    
    def predict(self, X, return_std=False):
        
        if return_std:
            self.mean, std= self.gp.predict(self.time.reshape(-1,1), return_std=True)
            # return self.mean, std + np.repeat(np.sqrt(np.diag(self.transportation_variance[:,:, 0])).reshape(-1,1), self.training_traj.shape[1], axis=1)
            return self.mean, np.repeat(np.sqrt(np.diag(self.transportation_variance[:,:, 0])).reshape(-1,1), self.training_traj.shape[1], axis=1)
        else:
            self.mean, _ = self.gp.predict(self.time.reshape(-1,1), return_std=True)
            return self.mean
    
    def samples(self, X, n_samples=10):
        # total_variance= self.gp.predict(self.time.reshape(-1,1), return_std=False,return_cov=True)[1] + self.transportation_variance
        total_variance= self.transportation_variance
        y_samples = [
            np.random.multivariate_normal(
                self.mean[:, target], total_variance[..., target], n_samples
            ).T[:, np.newaxis]
            for target in range(self.mean.shape[1])
        ]
        y_samples = np.hstack(y_samples)
        return y_samples
