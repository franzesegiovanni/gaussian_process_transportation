import tqdm
import torch
import gpytorch
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from  torch.autograd.functional import jacobian

x = torch.linspace(-4,4,1000).reshape(-1,1)
y = torch.cos(x) + torch.randn_like(x) * 0.2

train_n = len(x)
train_x = x[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = x[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

#plot the data
plt.scatter(train_x.cpu().numpy(), train_y.cpu().numpy(), label='Train')


from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)



from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.models import ExactGP

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
        K_inv=torch.linalg.inv(K)
        self.alpha=K_inv @ self.y_inducing
        self.R= torch.ones_like(self.var_inducing).cuda()- self.var_inducing
    
    def evaualte_kernel(self, x):
        kernel = self.covar_module(x,self.x_inducing)
        kernel= kernel.evaluate()
        kernel = torch.sum(kernel, dim=0)
        return kernel
    
    def jacobian_kernel(self, x):
        J= jacobian(self.evaualte_kernel, x)
        J= J.transpose(0,2) 
        return J
    def posterior_f(self, x, return_std=False):
        self.k_star=self.kernel(x,self.x_inducing)
        mu_exact= self.k_star @ self.alpha

        if return_std:
            
            k_start_start=self.kernel(x,x)
        
            sigma_exact=k_start_start -self.k_star @ self.K_inv @ self.k_star.t()
            sigma_exact= sigma_exact.evaluate()
            std_exact=torch.sqrt(torch.diag(sigma_exact)).reshape(-1,1)
            # std_exact_noise=torch.sqrt(torch.diag(sigma_exact_noise)).reshape(-1,1)
            return mu_exact,std_exact
    
        return mu_exact
   
    def kernel(self, x):
        return self.covar_module(x)

number_inducing_points = 5
inducing_points = torch.linspace(-2,2,number_inducing_points).reshape(-1,1)
model = GPModel(inducing_points=inducing_points)


if torch.cuda.is_available():
    model = model.cuda()



num_epochs = 100

model.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()}
], lr=0.01)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_y.size(0))


epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        y_batch=y_batch.reshape(-1,)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()

model.convert_to_exact_gp()

J= model.jacobian_kernel(train_x)