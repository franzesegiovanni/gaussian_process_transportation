import numpy as np
import time
from scipy.optimize import minimize
import scipy.optimize as sci_opti
from scipy.spatial import distance 

# This code was adapted from https://github.com/happygaoxiao/Diffeormorphism and modified to fit the needs of the project

# The original icea was formulated in the paper : "Perrin, Nicolas, and Philipp Schlehuber-Caissier. "Fast diffeomorphic matching to learn globally asymptotically stable nonlinear dynamical systems." Systems & Control Letters 96 (2016): 51-59."
class Iterative_Locally_Weighted_Translations():
    def __init__(self, para):
        # para: [k, rho, beta]
        # source: [n,7]
        # target: [n,7]
        self.k = int(para[0])
        self.para = para

    
    def fit(self, source, target):
        self.source = np.copy(source)
        self.target = np.copy(target)
        self.nums = source.shape[0]
        self.nbState = source.shape[1]  # dimension of position
        start_time = time.time()
        self.learnt_data = self.mapping(self.source, self.target)
        print ('Number of points =',self.nums,'in',self.nbState,'D space.')
        print ('iteration number =',self.k)
        print ('training time', time.time() - start_time, '[s].')
        self.mapping_error()
        
    def mapping(self, x, y):
        num_data = x.shape[0]
        # initialization
        k = self.k
        N = self.nbState
        rho_beta = np.zeros((k,2))
        p = np.zeros((k,N))   #  position transform center
        v = np.zeros((k,N))   #  transform vector
        self.source= np.copy(x)
        xi = np.copy(x)
        for i in range(k):
            # position iteration
            m = np.argmax(np.sum((xi - y)**2, axis=1))
            p[i,:] = xi[m, :]
            
            q = y[m,:]
            v0 = (q - p[i,:])
            # v0 = para[1]*(q - p[i,:])  # translation vector with the max distance, [2,]
            norm_v0 = np.sqrt(np.sum(v0**2))
            up_bound = self.para[1] *  np.sqrt(np.exp(1.)/2)/norm_v0 # for rho upbound to keep the diffeomorphism.
            x0 = np.array([[up_bound/10,0.5]])                 # initial values for [rho, beta]
            bnds = (0,up_bound),(0.9,0.9)  # rho, beta bounds
            args = (xi, v0, p[i,:], y, num_data)
            res_p = minimize(self.pos_cost_fun,x0.reshape(-1,),args,bounds=bnds) # solve the 2-parameter minimum problem
            rho_beta[i,:] = res_p.x
            v[i,:] = rho_beta[i,1] * v0
            xi = xi + np.exp(-rho_beta[i,0]**2 * np.sum((xi - p[i,:])**2,axis=1).reshape(-1,1))* v[i,:].reshape(1,-1)
        
        learnt_data = [rho_beta, p,v]
        return learnt_data 


    def predict(self, x, return_std=False):
        ### input:
        #           x: [n,3],  [x,y,z]
        ### output:
        #           y: [n,3],  [x,y,z]
        #           J:  [n, 3,3], Jacobian of forward mapping
        rho_beta, p, v = self.learnt_data[0], self.learnt_data[1], self.learnt_data[2]  # position data
        k = self.k
        y = np.copy(x)    
        sigma_total = np.zeros_like(y)
        for i in range(k):
            k_star= np.exp(-rho_beta[i,0]**2 * np.sum((y - p[i,:])**2,axis=1).reshape(-1,1))
            y = y+ k_star* v[i,:].reshape(1,-1)
            # this uncertainty is computed as each deformation was genegerate by a single point GP. At every iteratoin we sum the uncertainty from the previous steps. 
            sigma_iter = v[i,:]**2 * (1- k_star*k_star)
            sigma_total = sigma_total + sigma_iter
    
        if return_std:
            return y, np.zeros_like(y)
        else:
            return y

    
    def derivative(self, x, return_var=False):
        rho_beta, p, v = self.learnt_data[0], self.learnt_data[1], self.learnt_data[2]  # position data
        k = self.k
        N = self.nbState
        J = np.identity(N)
        J=np.repeat(J[np.newaxis, :, :], x.shape[0], axis=0)
        y = np.copy(x)       
        for i in range(k):
            tmp=np.exp(-rho_beta[i,0]**2 * np.sum((y - p[i,:])**2,axis=1).reshape(-1,1)) * (-2) * rho_beta[i,0]**2*(y-p[i,:])
            y = y+ np.exp(-rho_beta[i,0]**2 * np.sum((y - p[i,:])**2,axis=1).reshape(-1,1))* v[i,:].reshape(1,-1)
            v_i = v[i,:].reshape(-1,1)
            v_i= np.expand_dims(v_i,0)
            tmp = np.expand_dims(tmp,-2)
            J_i = np.identity(N) +  v_i @ tmp
            J = J_i @ J
        
        if return_var:
            return J, np.zeros_like(J)
        else:
            return J
    

    def pos_cost_fun(self,x, *args):
        xi, v0, p0, y, n = args[0],args[1],args[2],args[3],args[4]
        v0 = v0.reshape(1, -1)
        p0 = p0.reshape(1, -1)
        x22 = xi + x[1]*np.repeat(v0,n,axis=0)* np.exp(-x[0]**2 * np.sum((xi - p0)**2,axis=1).reshape(-1,1))
        dis = np.sum( np.sqrt(np.sum((x22-y)**2, axis=1)) )/n
        return dis

    
    def mapping_error(self):

        y_pred = self.predict(self.source)

        error = np.sqrt(np.sum((y_pred- self.target)**2,axis=1))
        print ('######### Estimation')
        print ('Total pos error mean+std:', np.mean(error), np.std(error), "[m]")

    def samples(self, X):
        # laplacian editing is deterministic, then we return the same sample
        predictions = [self.predict(X) for i in range(10)]
        predictions = np.array(predictions)  # Shape: (n_estimators, n_samples)
        return predictions
