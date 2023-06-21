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
from gpytorch.models import ApproximateGP
from  torch.autograd.functional import jacobian
# num_latents = 1
# num_tasks = 2

class SVGP(ApproximateGP):
    def __init__(self, X, Y, num_inducing=100, num_task=1, num_latents=1):
        # Let's use a different set of inducing points for each latent function

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        index=np.array(range(X.shape[0]))
        sample_index=np.random.choice(index, num_inducing)
        inducing_points= torch.from_numpy(X[sample_index,:]).float()

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_task,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]), ard_num_dims=X.shape[1], nu=1.5
        )
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Y.shape[1])
        self.X=torch.from_numpy(X).float()
        self.Y=torch.from_numpy(Y).float()

        # if torch.cuda.is_available():
        #     self.X= self.X.cuda()
        #     self.Y= self.Y.cuda()
        #     self.likelihood=self.likelihood.cuda()
        #     self.mean_module =self.mean_module.cuda()
        #     self.covar_module=self.covar_module.cuda()
        # self.mean_module=self.mean_module                                     
        # self.covar_module=self.covar_module
        # self.likelihood=self.likelihood
    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)       

class GaussianProcess():
    def __init__(self, X, Y, num_inducing=100):
        self.gp= SVGP(num_latents=1, X=X, Y=Y ,num_task=X.shape[1], num_inducing=num_inducing)

        train_dataset = TensorDataset(self.gp.X, self.gp.Y)
        self.train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    def fit(self, num_epochs=10):
        self.gp.train()
        self.gp.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.gp.parameters()},
        ], lr=0.01)

        self.mll = gpytorch.mlls.VariationalELBO(self.gp.likelihood, self.gp, num_data=self.gp.Y.size(0))
        # epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
        epochs_iter = tqdm(range(num_epochs))
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm(self.train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                output = self.gp(x_batch)
                #print(output)
                #print(y_batch)
                loss = -self.mll(output, y_batch)
                # print(loss)
                minibatch_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()  

    def mean_fun(self,x):
        predictions = self.gp.likelihood(self.gp(x))
        return predictions.mean
    def variance_fun(self,x):
        predictions = self.gp.likelihood(self.gp(x))
        return predictions.variance
    def predict(self,x):
        x=torch.from_numpy(x).float()
        predictions = self.gp.likelihood(self.gp(x))
        return predictions.mean.detach().numpy(), predictions.variance.detach().numpy()
         
    def derivative(self, x): 
        # jacobian(self.mean_fun, x)
        # print(type(x))
        x=torch.from_numpy(x).float()
        return jacobian(self.mean_fun, x).detach().numpy()#, jacobian(self.variance_fun, x).detach().numpy()
