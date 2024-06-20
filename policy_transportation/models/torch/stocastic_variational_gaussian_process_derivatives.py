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
from  torch.autograd.functional import jacobian
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

        # If you want to use different hyperparameters for each task,
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dim, batch_shape=batch_input),
            batch_shape=batch_output) # The scale kernel should be different for each task because they can have different unit of measure
        # interval=gpytorch.constraints.Interval(0.00000001,0.000001)        
        # self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_tasks, noise_constraint=interval)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_tasks)
    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def convert_to_exact_gp(self):
        self.x_inducing=self.variational_strategy.base_variational_strategy.inducing_points
        self.y_inducing=self.variational_strategy.base_variational_strategy.pseudo_points[1]
        self.var_inducing=self.variational_strategy.base_variational_strategy.pseudo_points[0]
        K=(self.covar_module(self.x_inducing,self.x_inducing)).evaluate()
        self.K_inv=torch.linalg.inv(K+ self.var_inducing )
        self.alpha=self.K_inv @ self.y_inducing
    
    def kernel_x_x_ind(self, x):
        kernel = self.covar_module(x,self.x_inducing)
        kernel= kernel.evaluate()
        kernel = torch.sum(kernel, dim=1)
        return kernel
    
    def kernel_10(self, x):
        J= jacobian(self.kernel_x_x_ind, x)
        J= J.permute(0,3,2,1)
        return J
    
    def kernel_output(self, x1, x2):   

        output = self.covar_module(x1,x2).evaluate()

        output = torch.sum(output, dim=1)
        return output

    def return_jacobian(self, x, y):
        inputs = (x, y)
        jac=jacobian(self.kernel_output, inputs, create_graph=True)
        jac_1= torch.sum(jac[0], dim=1)
        return jac_1
    
    def hessian(self,x,y):
        inputs = (x, y)
        hess= jacobian(self.return_jacobian, inputs)
        hessian= hess[1].permute(0,1,3,2,4)
        hessian=hessian.diagonal(dim1=-2, dim2=-1)
        hessian= hessian.permute(0,3,1,2)
        return hessian
    

    def posterior_f(self, x, return_std=False):
        self.k_star=self.covar_module(x,self.x_inducing)

        mu_exact= self.k_star @ self.alpha
        mu_exact= mu_exact.squeeze()
        if return_std:
            
            k_star_star=self.covar_module(x,x)
            sigma_exact = k_star_star - self.k_star @ self.K_inv @ self.k_star.transpose(-1,-2)
            sigma_exact= sigma_exact.evaluate()
            std_exact=torch.sqrt(sigma_exact.diagonal(dim1=-2,dim2=-1))
            mu_exact= mu_exact.squeeze()
            mu_exact= mu_exact.permute(1,0)
            std_exact= std_exact.permute(1,0)
            return mu_exact,std_exact
    
        return mu_exact
    

    def posterior_f_prime(self, x, return_std=False):

        kernel_10= self.kernel_10(x)#.squeeze()
        alpha = self.alpha.unsqueeze(1)
        mu_prime_exact= kernel_10 @ alpha
        mu_prime_exact= mu_prime_exact.squeeze()
        if return_std:
            x1= x.clone().detach().requires_grad_(True)
            x2= x.clone().detach().requires_grad_(True)
            k_star_star_prime=self.hessian(x1[0,:].reshape(1,-1), x2[0,:].reshape(1,-1)).squeeze(-1) # this works only if the kernel is statiionary
            rhs= kernel_10 @ self.K_inv @ kernel_10.transpose(-1,-2)
            rhs= rhs.squeeze()
            rhs_diag= rhs.diagonal(dim1=-2, dim2=-1)
            var_prime_exact = k_star_star_prime - rhs_diag
            std_prime_exact= torch.sqrt(var_prime_exact)

            # permute the output to be consistent with notation used in the paper
            mu_prime_exact= mu_prime_exact.permute(2,0,1)
            std_prime_exact= std_prime_exact.permute(2,0,1)
            return mu_prime_exact,std_prime_exact
    
        return mu_prime_exact
        
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
                # minibatch_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
        
        self.gp.convert_to_exact_gp()

    def predict(self,x, return_std=False):
        x=torch.from_numpy(x).float()
        if self.use_cuda:
            x=x.cuda()
        return self.gp.posterior_f(x, return_std=return_std)
         
    def derivative(self, x):
        x=torch.from_numpy(x).float()
        if self.use_cuda:
            x=x.cuda()
        J, J_std= self.gp.posterior_f_prime(x, return_std=True)
        return J, J_std