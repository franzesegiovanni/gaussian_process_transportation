import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from  torch.autograd.functional import jacobian, hessian


class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def convert_to_exact_gp(self):
        self.x_inducing=self.variational_strategy.inducing_points
        self.y_inducing=self.variational_strategy.pseudo_points[1]
        self.var_inducing=self.variational_strategy.pseudo_points[0]
        # self.kernel=self.covar_module
        K=(self.covar_module(self.x_inducing,self.x_inducing)).evaluate()
        self.K_inv=torch.linalg.inv(K)
        self.alpha=self.K_inv @ self.y_inducing
        self.R= K - self.var_inducing
    
    def evaualte_kernel(self, x):
        kernel = self.covar_module(x,self.x_inducing)
        kernel= kernel.evaluate()
        kernel = torch.sum(kernel, dim=0)
        return kernel
    
    def kernel_sum(self, x):
        kernel = self.covar_module(x,x)
        kernel= kernel.evaluate()
        kernel = torch.sum(kernel)
        return kernel
    
    def kernel_10(self, x):
        J= jacobian(self.evaualte_kernel, x)
        J= J.transpose(0,2) 
        return J.squeeze()
    def kernel_11(self, x):
        H= hessian(self.kernel_sum, x)
        return H.squeeze().squeeze()
    def posterior_f(self, x, return_std=False):
        self.k_star=self.covar_module(x,self.x_inducing)
        mu_exact= self.k_star @ self.alpha

        if return_std:
            
            k_start_start=self.covar_module(x,x)
            k_star_K_inv = self.k_star @ self.K_inv
            sigma_exact=k_start_start -k_star_K_inv @ self.R @ k_star_K_inv.t()
            sigma_exact= sigma_exact.evaluate()
            std_exact=torch.sqrt(torch.diag(sigma_exact)).reshape(-1,1)
            # std_exact_noise=torch.sqrt(torch.diag(sigma_exact_noise)).reshape(-1,1)
            return mu_exact,std_exact
    
        return mu_exact
    def posterior_f_prime(self, x, return_std=False):

        kernel_10= self.kernel_10(x)
        mu_exact= kernel_10 @ self.alpha
        if return_std:
            
            k_start_start=self.kernel_11(x)
        
            k_star_K_inv = kernel_10 @ self.K_inv
            sigma_exact=k_start_start -k_star_K_inv @ self.R @ k_star_K_inv.t()
            std_exact=torch.sqrt(torch.diag(sigma_exact)).reshape(-1,1)
            return mu_exact,std_exact
    
        return mu_exact
   
    def kernel(self, x):
        return self.covar_module(x)
    
class MultitaskGPModel(ApproximateGP):
    def __init__(self, num_tasks, inducing_points, batch_indipendent_kernel=False, ard=False):
        # Let's use a different set of inducing points for each latent function

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task

        if ard:
            ard_num_dim=inducing_points.size(-1)
        else:
            ard_num_dim=None

        if batch_indipendent_kernel:
            batch_shape=torch.Size([num_tasks])
        else:
            batch_shape=torch.Size([1])  

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            task_dim=-1
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_tasks]))

        # If you want to use different hyperparameters for each task,
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dim, nu=2.5, batch_shape=batch_shape),
            batch_shape=torch.Size([num_tasks])) # The scale kernel should be different for each task because they can have different unit of measure
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
        # self.kernel=self.covar_module
        K=(self.covar_module(self.x_inducing,self.x_inducing)).evaluate()
        self.K_inv=torch.linalg.inv(K)
        self.alpha=self.K_inv @ self.y_inducing
        self.R= K - self.var_inducing
    
    def evaualte_kernel(self, x):
        kernel = self.covar_module(x,self.x_inducing)
        kernel= kernel.evaluate()
        kernel = torch.sum(kernel, dim=1)
        return kernel
    
    def kernel_sum(self, x):
        kernel = self.covar_module(x,x)
        kernel= kernel.evaluate()
        kernel = torch.sum(kernel)
        return kernel
    
    def kernel_10(self, x):
        J= jacobian(self.evaualte_kernel, x)
        J= J.squeeze()
        J= J.transpose(-1,-2) 
        return J
    def kernel_11(self, x):
        H= hessian(self.kernel_sum, x)
        return H.squeeze().squeeze()
    def posterior_f(self, x, return_std=False):
        self.k_star=self.covar_module(x,self.x_inducing)
        mu_exact= self.k_star @ self.alpha
        mu_exact= mu_exact.squeeze()
        if return_std:
            
            k_start_start=self.covar_module(x,x)
            k_star_K_inv = self.k_star @ self.K_inv
            sigma_exact=k_start_start -k_star_K_inv @ self.R @ k_star_K_inv.transpose(-1,-2)
            sigma_exact= sigma_exact.evaluate()
            std_exact=torch.sqrt(sigma_exact.diagonal(dim1=-2,dim2=-1))
            mu_exact= mu_exact.squeeze()
            return mu_exact,std_exact
    
        return mu_exact
    def posterior_f_prime(self, x, return_std=False):

        kernel_10= self.kernel_10(x)
        mu_exact= kernel_10 @ self.alpha
        if return_std:
            
            k_start_start=self.kernel_11(x)
        
            k_star_K_inv = kernel_10 @ self.K_inv
            sigma_exact=k_start_start -k_star_K_inv @ self.R @ k_star_K_inv.t()
            std_exact=torch.sqrt(torch.diag(sigma_exact)).reshape(-1,1)
            return mu_exact,std_exact
    
        return mu_exact