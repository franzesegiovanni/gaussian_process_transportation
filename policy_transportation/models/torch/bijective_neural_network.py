import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm 
from torch import autograd
import torch.nn.functional as F
import numpy as np
import random

class BiJectiveNetwork():
    def __init__(self, X, Y):
        seed = random.randint(1, 10000)

		# Set random seed for PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        self.use_cuda = torch.cuda.is_available()
        self.use_cuda=False
        num_blocks = 4                 # number of coupling layers
        num_hidden = 20                        # hidden layer dimensions (there are two of hidden layers)
        # only for fcnn!
        t_act = 'elu'                           # activation fcn in each network (must be continuously differentiable!)
        s_act = 'elu'
        input_size=X.shape[1]
        self.nn= BijectionNet(num_dims=input_size, num_blocks=num_blocks, num_hidden=num_hidden, s_act=s_act, t_act=t_act)
        if self.use_cuda:
            self.nn=self.nn.cuda()
        X=torch.from_numpy(X).float()
        Y=torch.from_numpy(Y).float()
        train_dataset = TensorDataset(X, Y)
        self.train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True) 


    def fit(self, num_epochs=100):
        # Define the loss function and optimizer

        criterion = nn.SmoothL1Loss()  # Mean Squared Error for regression
        optimizer = optim.Adam(self.nn.parameters(), lr=0.001)

    
        # Training loop
        epochs_iter = tqdm(range(num_epochs))
        for epoch in epochs_iter:
            for x_batch, y_batch in tqdm(self.train_loader, desc="Minibatch", leave=False):
                # Forward pass
                if self.use_cuda:
                    x_batch=x_batch.cuda()
                    y_batch=y_batch.cuda()
                outputs = self.nn(x_batch)[0]
                loss = criterion(outputs, y_batch)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
             
    def derivative(self, x): 
        x=torch.from_numpy(x).float()
        if self.use_cuda:
            x=x.cuda()
        J = self.nn(x)[1]
        return J.detach().numpy()
    
    def predict(self, x): 
        x=torch.from_numpy(x).float()
        if self.use_cuda:
            x=x.cuda()
        predictions = self.nn(x)[0]
        return predictions.detach().cpu().numpy()
         
class BijectionNet(nn.Sequential):
	"""
	A sequential container of flows based on coupling layers.
	"""
	def __init__(self, num_dims, num_blocks, num_hidden, s_act=None, t_act=None, sigma=None,
				 coupling_network_type='fcnn'):
		self.num_dims = num_dims
		modules = []
		# print('Using the {} for coupling layer'.format(coupling_network_type))
		mask = torch.arange(0, num_dims) % 2  # alternating inputs
		mask = mask.float()
		# mask = mask.to(device).float()
		for _ in range(num_blocks):
			modules += [
				CouplingLayer(
					num_inputs=num_dims, num_hidden=num_hidden, mask=mask,
					s_act=s_act, t_act=t_act, sigma=sigma, base_network=coupling_network_type),
			]
			mask = 1 - mask  # flipping mask
		super(BijectionNet, self).__init__(*modules)

	def jacobian(self, inputs, mode='direct'):
		'''
		Finds the product of all jacobians
		'''
		batch_size = inputs.size(0)
		J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

		if mode == 'direct':
			for module in self._modules.values():
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				# inputs = module(inputs, mode)
		else:
			for module in reversed(self._modules.values()):
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				# inputs = module(inputs, mode)
		return J

	def forward(self, inputs, mode='direct'):
		""" Performs a forward or backward pass for flow modules.
		Args:
			inputs: a tuple of inputs and logdets
			mode: to run direct computation or inverse
		"""
		assert mode in ['direct', 'inverse']
		batch_size = inputs.size(0)
		J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

		if mode == 'direct':
			for module in self._modules.values():
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				inputs = module(inputs, mode)
		else:
			for module in reversed(self._modules.values()):
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				inputs = module(inputs, mode)
		return inputs, J        
        
    
class CouplingLayer(nn.Module):
	""" An implementation of a coupling layer
	from RealNVP (https://arxiv.org/abs/1605.08803).
	"""

	def __init__(self, num_inputs, num_hidden, mask,
				 base_network='rffn', s_act='elu', t_act='elu', sigma=0.45):
		super(CouplingLayer, self).__init__()

		self.num_inputs = num_inputs
		self.mask = mask

		if base_network == 'fcnn':
			self.scale_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=s_act)
			self.translate_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=t_act)
			# print('Using neural network initialized with identity map!')

			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.translate_net.network[-1].bias.data)

			nn.init.zeros_(self.scale_net.network[-1].weight.data)
			nn.init.zeros_(self.scale_net.network[-1].bias.data)

		elif base_network == 'rffn':
			print('Using random fouier feature with bandwidth = {}.'.format(sigma))
			self.scale_net = RFFN(in_dim=num_inputs, out_dim=num_inputs, nfeat=num_hidden, sigma=sigma)
			self.translate_net = RFFN(in_dim=num_inputs, out_dim=num_inputs, nfeat=num_hidden, sigma=sigma)

			# print('Initializing coupling layers as identity!')
			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.scale_net.network[-1].weight.data)
		else:
			raise TypeError('The network type has not been defined')

	def forward(self, inputs, mode='direct'):
		mask = self.mask
		masked_inputs = inputs * mask
		# masked_inputs.requires_grad_(True)

		log_s = self.scale_net(masked_inputs) * (1 - mask)
		t = self.translate_net(masked_inputs) * (1 - mask)

		if mode == 'direct':
			s = torch.exp(log_s)
			return inputs * s + t
		else:
			s = torch.exp(-log_s)
			return (inputs - t) * s

	def jacobian(self, inputs):
		return get_jacobian(self, inputs, inputs.size(-1))

class RFFN(nn.Module):
	"""
	Random Fourier features network.
	"""

	def __init__(self, in_dim, out_dim, nfeat, sigma=10.):
		super(RFFN, self).__init__()
		self.sigma = np.ones(in_dim) * sigma
		self.coeff = np.random.normal(0.0, 1.0, (nfeat, in_dim))
		self.coeff = self.coeff / self.sigma.reshape(1, len(self.sigma))
		self.offset = 2.0 * np.pi * np.random.rand(1, nfeat)

		self.network = nn.Sequential(
			LinearClamped(in_dim, nfeat, self.coeff, self.offset),
			Cos(),
			nn.Linear(nfeat, out_dim, bias=False)
		)

	def forward(self, x):
		return self.network(x)


class FCNN(nn.Module):
	'''
	2-layer fully connected neural network
	'''

	def __init__(self, in_dim, out_dim, hidden_dim, act='tanh'):
		super(FCNN, self).__init__()
		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
					   'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus}

		act_func = activations[act]
		self.network = nn.Sequential(
			nn.Linear(in_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, out_dim)
		)

	def forward(self, x):
		return self.network(x)


class LinearClamped(nn.Module):
	'''
	Linear layer with user-specified parameters (not to be learrned!)
	'''

	__constants__ = ['bias', 'in_features', 'out_features']

	def __init__(self, in_features, out_features, weights, bias_values, bias=True):
		super(LinearClamped, self).__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.register_buffer('weight', torch.Tensor(weights))
		if bias:
			self.register_buffer('bias', torch.Tensor(bias_values))

	def forward(self, input):
		if input.dim() == 1:
			return F.linear(input.view(1, -1), self.weight, self.bias)
		return F.linear(input, self.weight, self.bias)

	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)


class Cos(nn.Module):
	"""
	Applies the cosine element-wise function
	"""

	def forward(self, inputs):
		return torch.cos(inputs)


def get_jacobian(net, x, output_dims, reshape_flag=True):
	"""

	"""
	if x.ndimension() == 1:
		n = 1
	else:
		n = x.size()[0]
	x_m = x.repeat(1, output_dims).view(-1, output_dims)
	x_m.requires_grad_(True)
	y_m = net(x_m)
	mask = torch.eye(output_dims).repeat(n, 1).to(x.device)
	# y.backward(mask)
	J = autograd.grad(y_m, x_m, mask, create_graph=True)[0]
	if reshape_flag:
		J = J.reshape(n, output_dims, output_dims)
	return J        
