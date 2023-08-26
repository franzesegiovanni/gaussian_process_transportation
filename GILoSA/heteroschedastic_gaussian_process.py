"""
Authors: Ravi Prakash & Giovanni Franzese, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
class HeteroschedasticGaussianProcess():
    def __init__(self, kernel, alpha=1e-10, n_restarts_optimizer=5):
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)
        self.kernel=kernel
        self.alpha=alpha
    def fit(self, X, Y, sigma_noise): #sigma_noise is the heteroschedastic noise
            self.X=X
            self.Y=Y
            self.n_features=np.shape(self.X)[1]
            self.n_samples=np.shape(self.X)[0]
            self.gp.fit(self.X,self.Y)
            self.kernel= self.gp.kernel_
            self.kernel_params_= [self.kernel.get_params()['k1__k2__length_scale'], self.kernel.get_params()['k1']]
            self.noise_var_ = self.gp.alpha + self.kernel.get_params()['k2__noise_level']
            self.max_var   = self.kernel.get_params()['k1__k1__constant_value']+ self.noise_var_
            K_ = self.kernel(self.X, self.X) #+ (self.noise_var_ * np.eye(len(self.X)))
            K_plus_D= K_ + np.diag(sigma_noise)
            self.K_inv = np.linalg.inv(K_)
            self.K_plus_D_inv = np.linalg.inv(K_plus_D)
            print('lenghtscales', self.kernel.get_params()['k1__k2__length_scale'] )

    def predict(self, x, sigma_noise=None, return_var=False):

        k_star = self.kernel(x, self.X)
        #.reshape(-1, 1)
        # print('k_star.shape')
        # print(k_star.shape)
        # print('self.K_plus_D_inv.shape')
        # print(self.K_plus_D_inv.shape)
        # print('self.Y.shape')
        # print(self.Y.shape)
        k_star_K_inv_ = np.matmul(k_star, self.K_plus_D_inv)
        self.mu=np.matmul(k_star_K_inv_, self.Y)
        self.sigma=None
        if return_var==True:
            # print(sigma_noise)
            self.sigma = self.kernel(x, x) - k_star_K_inv_ @ k_star.transpose()  + np.diag(sigma_noise) #+ self.noise_var_ 
        return self.mu, self.sigma 
         
    def derivative(self, x): # GP with RBF kernel 
        """Input has shape n_query x n_features. 
        There are two outputs,
        1. derivative of the mean
        2. derivative of the predicted variance 
        Each utput has shape n_query x n_features x n_outputs.
        The output in position i,j,k has the derivative respect to the j-th feature of the k-th output, in position of the i-th data point.
        For the derivative of sigma n_outputs is equal to 1"""
        lscale=self.kernel_params_[0].reshape(-1,1)
        lscale_stack= np.hstack([lscale]*self.n_samples)
        alfa= np.matmul(self.K_plus_D_inv, self.Y)
        dy_dx=[]
        dsigma_dx=[]
        for i in range(np.shape(x)[0]):
            k_star= self.kernel(self.X, x[i,:].reshape(1,-1))
            k_star_T=k_star.transpose()
            k_star_stack= np.vstack([k_star_T]*self.n_features)
            beta = -2 * np.matmul(self.K_plus_D_inv, k_star)
            dk_star_dX= k_star_stack * (self.X- x[i,:]).transpose()/ (lscale_stack** 2)
            dy_dx.append(np.matmul(dk_star_dX, alfa))
            dsigma_dx.append(np.matmul(dk_star_dX, beta))
        return np.array(dy_dx), np.array(dsigma_dx)