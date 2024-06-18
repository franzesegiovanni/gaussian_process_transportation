import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from  torch.autograd.functional import jacobian, hessian

class MultitaskGPModel(ApproximateGP):
    def __init__(self, num_tasks, inducing_points, batch_indipendent_kernel=False, ard=False):
        # Let's use a different set of inducing points for each latent function

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        self.num_tasks = num_tasks
        self.num_inputs= inducing_points.size(-1)
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

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)

        # If you want to use different hyperparameters for each task,
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dim, batch_shape=batch_shape),
            batch_shape=batch_shape) # The scale kernel should be different for each task because they can have different unit of measure
    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def convert_to_exact_gp(self):
        # expected_max_var_derivative=self.covar_module.outputscale/self.covar_module.base_kernel.lengthscale**2
        # print("expected_max_var_derivative")
        # print(expected_max_var_derivative)
        # hess=self.hessian(torch.zeros([1, self.num_inputs]).cuda(), torch.zeros([1, self.num_inputs]).cuda()).squeeze()
        # print("hessian")
        # print(hess)
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
        J= J.transpose(-1,-3)
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
            return mu_exact,std_exact
    
        return mu_exact
    def posterior_f_prime(self, x, return_std=False):

        kernel_10= self.kernel_10(x).squeeze()
        mu_prime_exact= kernel_10 @ self.alpha
        mu_prime_exact= mu_prime_exact.squeeze()
        if return_std:
            x1= x.clone().detach().requires_grad_(True)
            x2= x.clone().detach().requires_grad_(True)
            k_star_star_prime=self.hessian(x1[0,:].reshape(-1,1), x2[0,:].reshape(-1,1)).reshape(-1,1)
            rhs= kernel_10 @ self.K_inv @ kernel_10.transpose(-1,-2)
            rhs= rhs.squeeze()
            rhs_diag= rhs.diagonal(dim1=-2, dim2=-1)
            var_prime_exact = k_star_star_prime - rhs_diag
            std_prime_exact= torch.sqrt(var_prime_exact)
            return mu_prime_exact,std_prime_exact
    
        return mu_prime_exact