"""
Authors: Giovanni Franzese & Ravi, March 2023
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
class GaussianProcess():
    def __init__(self, kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, n_targets=None):
        if optimizer != None:
            self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=optimizer ,n_restarts_optimizer=n_restarts_optimizer, n_targets=n_targets)
        else:
            self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=optimizer, n_targets=n_targets)      
        self.kernel=kernel
        self.alpha=alpha
    def fit(self, X, Y):
            self.X=X
            self.Y=Y
            self.n_features=np.shape(self.X)[1]
            self.n_samples=np.shape(self.X)[0]
            self.n_outputs=np.shape(self.Y)[1]
            self.gp.fit(self.X,self.Y)
            self.kernel= self.gp.kernel_
            self.kernel_params_= [self.kernel.get_params()['k1__k2__length_scale'], self.kernel.get_params()['k1']]
            self.noise_var_ = self.gp.alpha + self.kernel.get_params()['k2__noise_level']
            self.prior_var   = self.kernel.get_params()['k1__k1__constant_value']
            K_ = self.kernel(self.X, self.X) + (self.noise_var_ * np.eye(len(self.X)))
            self.K_inv = np.linalg.inv(K_)
            print('lenghtscales', self.kernel.get_params()['k1__k2__length_scale'] )

    def predict(self,x, return_std=True, return_cov=False):
        if return_std==True:
            [y, std]=self.gp.predict(x, return_std=return_std)
            return np.array(y), np.array(std-np.sqrt(self.kernel.get_params()['k2__noise_level']))
        if return_cov==True:
            [y, cov]=self.gp.predict(x, return_cov=return_cov)
            return np.array(y), np.array(cov)
        else:
            y=self.gp.predict(x, return_std=return_std)
            return np.array(y)
    
    def samples(self, x):
        samples=self.gp.sample_y(x, n_samples=10)
        samples_transpose=np.transpose(samples, (2, 0, 1))
        return samples_transpose
    
    
    def derivative(self, x, return_var=False): # here we predict p(f'*| f, y). 
        """Input has shape n_query x n_features. 
        There are two outputs,
        1. mean of the derivative function 
        2. predicted standar deviation of the first derivative
        Each output has shape  batch_dimension x n_features x n_outputs.
        The output in position i,j,k has the derivative respect to the k-th feature of the j-th output, in position of the i-th data point.
        The uncertinaty is the variance of each element of the Jacobian matrix.
        """
        lscale=self.kernel_params_[0].reshape(-1,1)
        alfa=  self.K_inv @ self.Y 
        k_star= self.kernel(x, self.X)
        X_T= (self.X).transpose()
        x_T = x.transpose()
        X_reshaped = X_T[:,  np.newaxis,:]
        x_reshaped = x_T[:,  :, np.newaxis]
        lascale_rehaped= lscale[ :,  :, np.newaxis]

        # Calculate the difference
        difference_matrix =  X_reshaped - x_reshaped
        
        coefficient= difference_matrix/ ( lascale_rehaped** 2) 

        #coefficient of dk_dx_prime that multiplies the kernel itself  (x - x_prime)/sigma_l**2
        dk_star_dx=  coefficient * k_star
        dk_star_dx_transpose= dk_star_dx.transpose(1,0,2)
        df_dx = dk_star_dx_transpose @ alfa 
        df_dx= df_dx.transpose(0,2,1)

        if return_var==True:
            #dk2_dx_dx_prime : Sigma_v/sigma_l**2
            # dk_star_dx_transpose= dk_star_dx.transpose(1,0,2)
            dk_star_dx_K_inv_= dk_star_dx @  self.K_inv

            diag_k_K_inv_k = np.sum(dk_star_dx_K_inv_ * dk_star_dx, axis=2)
            var= self.prior_var/(lscale**2) - diag_k_K_inv_k
            Sigma_df_dx= np.repeat(var[np.newaxis, :,:], self.n_outputs, axis=0)
            Sigma_df_dx= Sigma_df_dx.transpose(2,0,1)
            return df_dx, Sigma_df_dx
        return df_dx 
    
    def derivative_of_variance(self,x):
        lscale=self.kernel_params_[0].reshape(-1,1)
        k_star= self.kernel(x, self.X)
        X_T= (self.X).transpose()
        x_T = x.transpose()
        X_reshaped = X_T[:,  np.newaxis,:]
        x_reshaped = x_T[:,  :, np.newaxis]
        lascale_rehaped= lscale[ :,  :, np.newaxis]

        # Calculate the difference
        difference_matrix =  X_reshaped - x_reshaped
        
        coefficient= difference_matrix/ ( lascale_rehaped** 2) 

        #coefficient of dk_dx_prime that multiplies the kernel itself  (x - x_prime)/sigma_l**2
        dk_star_dx=  coefficient * k_star

        #dk2_dx_dx_prime : Sigma_v/sigma_l**2
        # dk_star_dx_transpose= dk_star_dx.transpose(1,0,2)
        dk_star_dx_K_inv_= dk_star_dx @  self.K_inv

        diag_k_K_inv_k = np.sum(dk_star_dx_K_inv_ * dk_star_dx, axis=2)
        return diag_k_K_inv_k

