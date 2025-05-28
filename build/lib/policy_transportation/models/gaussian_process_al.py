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
class GaussianProcess():
    def __init__(self, kernel, alpha=1e-10, n_restarts_optimizer=5, n_samples_max=20000):
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)
        self.kernel=kernel
        self.alpha=alpha
        self.n_samples_max=n_samples_max
    def fit(self, X, Y):
            self.X=X
            self.Y=Y
            self.n_features=np.shape(self.X)[1]
            self.n_samples=np.shape(self.X)[0]
            if self.n_samples > self.n_samples_max:
                print("Starting Active Learning")
                n_initial = int(0.1*self.n_samples_max)
                X_tmp=np.copy(self.X)
                Y_tmp=np.copy(self.Y)
                initial_idx = np.random.choice(range(self.n_samples), size=n_initial, replace=False)
                X_sample= X_tmp[initial_idx]
                Y_sample = Y_tmp[initial_idx]
                X_tmp=np.delete(X_tmp, initial_idx, axis=0)
                Y_tmp=np.delete(Y_tmp,initial_idx, axis=0)
                gp_active = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
                gp_active.fit(X_sample,Y_sample)
                self.n_samples_batch=int(self.n_samples_max/20) 
                for i in tqdm(range(int((self.n_samples_max-n_initial)))):
                    #print("Added sample number",n_initial + i+1, "out of", self.n_samples_max)
                    [mean, std]=gp_active.predict(X_tmp, return_std=True)
                    print("std.shape")
                    print(std.shape) 
                    print("Y_tmp.shape[1]")
                    print(Y_tmp.shape[1]) 
                    std=std.reshape(-1,Y_tmp.shape[1])
                    query_idx = np.argmax(std[:,0])
                    #indices = np.argsort(std[:,0])  # Get the indices that would sort the array
                    #query_idx = indices[-self.n_samples_batch:]

                    X_sample=np.vstack([X_sample, X_tmp[query_idx]])
                    Y_sample=np.vstack([Y_sample, Y_tmp[query_idx]])
                    X_tmp=np.delete(X_tmp, query_idx, axis=0)
                    Y_tmp=np.delete(Y_tmp,query_idx, axis=0)
                    # add the point with the maximun error also:
                    # print("X_sample",X_sample)
                    gp_active.fit(X_sample,Y_sample)
                [mean, std]=gp_active.predict(self.X, return_std=True)
                error= np.mean(np.sum(np.abs(mean-self.X), axis=1))
                print("error:", error)
            # query_idx = np.argmax(error)
            # X_sample=np.vstack([X_sample, self.X[query_idx]])
            # Y_sample=np.vstack([Y_sample, self.Y[query_idx]])

                self.n_samples=self.n_samples_max    
                self.X=np.copy(X_sample)
                self.Y =np.copy(Y_sample) 
            self.gp.fit(self.X,self.Y)
            self.kernel= self.gp.kernel_
            self.kernel_params_= [self.kernel.get_params()['k1__k2__length_scale'], self.kernel.get_params()['k1']]
            self.noise_var_ = self.gp.alpha + self.kernel.get_params()['k2__noise_level']
            self.max_var   = self.kernel.get_params()['k1__k1__constant_value']+ self.noise_var_
            K_ = self.kernel(self.X, self.X) + (self.noise_var_ * np.eye(len(self.X)))
            self.K_inv = np.linalg.inv(K_)
            print('lenghtscales', self.kernel.get_params()['k1__k2__length_scale'] )
    def predict(self,x):
        [y, std]=self.gp.predict(x, return_std=True)
        
        return np.array(y), np.array(std-np.sqrt(self.kernel.get_params()['k2__noise_level']))
         
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
        alfa= np.matmul(self.K_inv, self.Y)
        dy_dx=[]
        dsigma_dx=[]
        for i in range(np.shape(x)[0]):
            k_star= self.kernel(self.X, x[i,:].reshape(1,-1))
            k_star_T=k_star.transpose()
            k_star_stack= np.vstack([k_star_T]*self.n_features)
            beta = -2 * np.matmul(self.K_inv, k_star)
            # print(self.X.shape)
            # print(x[i,:].shape)
            # print(k_star_stack.shape)
            # a=(self.X- x[i,:])
            # print(a.shape)
            dk_star_dX= k_star_stack * (self.X- x[i,:]).transpose()/ (lscale_stack** 2)
            dy_dx.append(np.matmul(dk_star_dX, alfa))
            dsigma_dx.append(np.matmul(dk_star_dX, beta))
        return np.array(dy_dx), np.array(dsigma_dx)
