import networkx as nx
import numpy as np
from policy_transportation import GaussianProcess as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
class KMP():
    def __init__(self):
        pass

    def fit(self, time, training_traj, kernel=C(0.1, constant_value_bounds=[0.1,  10]) * RBF(length_scale=[0.1], length_scale_bounds=[0.3, 1]) + WhiteKernel(0.00001)):
        self.training_traj=training_traj
        # we define the normalized time from 0 to 1
        self.time=time
        self.gp=GPR(kernel, n_targets=self.training_traj.shape[1])
        self.gp.fit(self.time, self.training_traj)
        # We fit the movement primitives x= GP(t) to the training trajectory

        self.noise_var_ = self.gp.alpha + self.gp.kernel.get_params()['k2__noise_level']
    
    def correct(self, t_star, y_star):
        '''
        Correct the movement primitives with the difference between the target and the source distribution

        Parameters
        ----------
        t_star : array of time of the trajectory that we want to change

        y_star : array of the new trajectory points we want to have at a certain time
        '''
        k_star= self.gp.kernel(self.time, t_star)
        K_star_star = self.gp.kernel(t_star, t_star)
        K_star_star_noise = K_star_star + np.eye(K_star_star.shape[0])*self.noise_var_
        self.noise_var_ = self.gp.alpha + self.gp.kernel.get_params()['k2__noise_level']
        self.training_traj = self.training_traj + k_star @ np.linalg.inv(K_star_star_noise) @ (y_star- self.gp.predict(t_star))
        print("Fitting the transportation function")
        self.gp.fit(self.time, self.training_traj)

        self.transportation_variance = self.gp.kernel(self.time, self.time) - k_star @ np.linalg.inv(K_star_star_noise) @ k_star.T

        self.transportation_variance = np.repeat(self.transportation_variance[:,:,np.newaxis], self.training_traj.shape[1], axis=2)

    def predict(self, time, return_std=False):
        
        if return_std:
            self.mean= self.gp.predict(time)
            return self.mean, np.repeat(np.sqrt(np.diag(self.transportation_variance[:,:, 0])).reshape(-1,1), self.training_traj.shape[1], axis=1)
        else:
            self.mean, _ = self.gp.predict(time, return_std=True)
            return self.mean
        
    def derivative(self, time, return_var=False):
        return self.gp.derivative(time, return_var=return_var)
    
    def samples(self, X, n_samples=10):
        total_variance= self.transportation_variance
        y_samples = [
            np.random.multivariate_normal(
                self.mean[:, target], total_variance[..., target], n_samples
            ).T[:, np.newaxis]
            for target in range(self.mean.shape[1])
        ]
        y_samples = np.hstack(y_samples)
        return y_samples
