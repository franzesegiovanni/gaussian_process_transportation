"""
Authors: Giovanni Franzese, March 2023
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import numpy as np
from tqdm import tqdm 
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from  torch.autograd.functional import jacobian, hessian as Hessian
from torch.autograd import grad
from gpytorch.models import ApproximateGP

class SVGP(ApproximateGP):
    def __init__(self,X, Y, num_inducing=200, ard=True):
        # Let's use a different set of inducing points for each task
        self.X=torch.from_numpy(X).float()
        self.Y=torch.from_numpy(Y).float()
        index=np.array(range(X.shape[0]))
        sample_index=np.random.choice(index, num_inducing)
        self.inducing_points= torch.from_numpy(X[sample_index,:]).float()
        self.inducing_outputs= torch.from_numpy(Y[sample_index,:]).float()
        # Let's use a different set of inducing points for each latent function

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        self.num_tasks = self.Y.size(-1)
        self.num_inputs= self.inducing_points.size(-1)
        if ard:
            ard_num_dim=self.inducing_points.size(-1)
        else:
            ard_num_dim=None

        
        batch_output=torch.Size([self.num_tasks])
        batch_input=torch.Size([1]) 


        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            self.inducing_points.size(-2), batch_shape=torch.Size([self.num_tasks])
        )

        variational_distribution.variational_mean.data= self.inducing_outputs.t() 

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, self.inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=self.num_tasks,
            task_dim=-1
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_output)

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dim, batch_shape=batch_input), batch_shape=batch_output)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_tasks)
    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def convert_to_exact_gp(self):
        self.x_inducing=self.variational_strategy.base_variational_strategy.inducing_points
        # self.y_inducing=self.variational_strategy.base_variational_strategy.pseudo_points[1]
        # self.var_inducing=self.variational_strategy.base_variational_strategy.pseudo_points[0]
        # K=(self.covar_module(self.x_inducing,self.x_inducing)).evaluate()
        # self.K_inv=torch.linalg.inv(K+ self.var_inducing )
        # self.alpha=self.K_inv @ self.y_inducing

        m=self.variational_strategy.base_variational_strategy._variational_distribution.variational_mean
        S_cho=self.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar
        S = S_cho @ S_cho.transpose(-1,-2)
        x_inducing=self.variational_strategy.base_variational_strategy.inducing_points
        kernel=self.covar_module
        self.K=(kernel(x_inducing,x_inducing)).evaluate()
        L=torch.linalg.cholesky(self.K)
        self.real_m= L @ m.unsqueeze(-1)
        self.real_S  = L @ S @ S.transpose(-1,-2) @ L.transpose(-1,-2)
        self.K_inv=torch.linalg.inv(self.K)
        self.alpha=self.K_inv @ self.real_m
        self.K_inv_svgp = self.K_inv @ (self.K-self.real_S) @ self.K_inv.transpose(-1,-2)

    
    def kernel_x_x_ind(self, x):
        kernel = self.covar_module(x,self.x_inducing)
        kernel= kernel.evaluate()
        kernel = torch.sum(kernel, dim=1)
        return kernel
    
    def kernel_10(self, x):
        J= jacobian(self.kernel_x_x_ind, x)
        J= J.permute(0,3,2,1)
        return J

    def kernel_11(self,x):
        if self.covar_module.is_stationary:
            x_1=x[0].clone().detach().requires_grad_(True).reshape(1,-1)
            x_2=x[0].clone().detach().requires_grad_(True).reshape(1,-1)
            k=self.covar_module(x_1,x_2).evaluate().squeeze()
            HESS = torch.empty(( k.shape[0],x_1.shape[1]))
            for i in range(k.shape[0]):
                grad_i = grad(k[i], x_1, retain_graph=True, create_graph=True)[0].reshape(-1)
                for j in range(grad_i.shape[0]):
                    HESS[i,j] = grad(grad_i[j], x_2,retain_graph=True, create_graph=True)[0][0][j]
            #copy the hessian as a batch of size x.shape[0]
            HESS= HESS.unsqueeze(-1).repeat(1,1,x.shape[0])
            return HESS.cuda()
        else:
            raise ValueError("The kernel is not stationary, the derivative is not implemented yet")

    def posterior_f(self, x, return_std=False):
        self.k_star=self.covar_module(x,self.x_inducing)

        mu_exact= self.k_star @ self.alpha
        mu_exact= mu_exact.squeeze()
        if return_std:
            
            k_star_star=self.covar_module(x,x)
            sigma_exact = k_star_star - self.k_star @ self.K_inv_svgp @ self.k_star.transpose(-1,-2)
            sigma_exact= sigma_exact.evaluate()
            std_exact=torch.sqrt(sigma_exact.diagonal(dim1=-2,dim2=-1))
            mu_exact= mu_exact.squeeze()
            mu_exact= mu_exact.permute(1,0)
            std_exact= std_exact.permute(1,0)
            return mu_exact.detach().cpu().numpy(), std_exact.cpu().detach().numpy() 
        else:
            return mu_exact.detach().cpu().numpy()
    

    def posterior_f_prime(self, x, return_var=False):

        kernel_10= self.kernel_10(x)
        alpha = self.alpha.unsqueeze(1)
        mu_prime_exact= kernel_10 @ alpha
        mu_prime_exact= mu_prime_exact.squeeze()
        if return_var:
            k_star_star_prime=self.kernel_11(x) # this works only if the kernel is statiionary
            rhs= kernel_10 @ self.K_inv_svgp @ kernel_10.transpose(-1,-2)
            rhs= rhs.squeeze()
            rhs_diag= rhs.diagonal(dim1=-2, dim2=-1)
            var_prime_exact = k_star_star_prime - rhs_diag
            # permute the output to be consistent with notation used in the paper
            mu_prime_exact= mu_prime_exact.permute(2,0,1)
            var_prime_exact= var_prime_exact.permute(2,0,1)
            return mu_prime_exact.detach().cpu().numpy(), var_prime_exact.cpu().detach().numpy()
        else:
            mu_prime_exact= mu_prime_exact.permute(2,0,1)
            return mu_prime_exact.detach().cpu().numpy()
        
class StocasticVariationalGaussianProcess():
    def __init__(self, X, Y, num_inducing=100):
        torch.cuda.empty_cache()
        self.use_cuda=torch.cuda.is_available()
        self.gp= SVGP(X, Y, num_inducing=num_inducing)
        if self.use_cuda:
            self.gp=self.gp.cuda()
        train_dataset = TensorDataset(self.gp.X, self.gp.Y)
        self.train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)    
    def fit(self, num_epochs=10):
        self.gp.train()
        self.gp.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.gp.parameters()},
        ], lr=0.01)

        self.mll = gpytorch.mlls.VariationalELBO(self.gp.likelihood, self.gp, num_data=self.gp.Y.size(0))
        epochs_iter = tqdm(range(num_epochs))
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = self.train_loader
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                if self.use_cuda:
                    x_batch=x_batch.cuda()
                    y_batch=y_batch.cuda()
                output = self.gp(x_batch)
                loss = -self.mll(output, y_batch)
                loss.backward()
                optimizer.step()
        # print vertical and horizzontal length scale
        print("Output length scale: ", self.gp.covar_module.outputscale)
        print("Horizontal length scale: ", self.gp.covar_module.base_kernel.lengthscale)
        
        self.gp.convert_to_exact_gp()

    def predict(self,x, return_std=False):
        x=torch.from_numpy(x).float()
        if self.use_cuda:
            x=x.cuda()
        return self.gp.posterior_f(x, return_std=return_std)
         
    def derivative(self, x, return_var=False):
        x=torch.from_numpy(x).float()
        if self.use_cuda:
            x=x.cuda()
        if return_var:
            return self.gp.posterior_f_prime(x, return_var=True)    
        else:
            return self.gp.posterior_f_prime(x, return_var=False)