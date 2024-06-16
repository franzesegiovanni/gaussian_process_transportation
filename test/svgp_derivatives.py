import tqdm
import torch
import gpytorch
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from models import GPModel
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


number_inducing_points = 5
inducing_points = torch.linspace(-2,2,number_inducing_points).reshape(-1,1)
model = GPModel(inducing_points=inducing_points)


if torch.cuda.is_available():
    model = model.cuda()



num_epochs = 1000

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

# J= model.kernel_10(train_x)

# H = model.kernel_11(train_x)

posterior_f, std_f=model.posterior_f(train_x, return_std=True)
posterior_f_prime, std_f_prime=model.posterior_f_prime(train_x, return_std=True)

#plot the data and the derivatives
with torch.no_grad():
    plt.scatter(train_x.cpu().numpy(), train_y.cpu().numpy(), label='Train')
    plt.plot(train_x.cpu().numpy(), posterior_f.cpu().numpy(), label='Posterior f')
    plt.fill_between(train_x.cpu().numpy().reshape(-1), (posterior_f-std_f).cpu().numpy().reshape(-1), (posterior_f+std_f).cpu().numpy().reshape(-1), alpha=0.5)

    plt.plot(train_x.cpu().numpy(), posterior_f_prime.cpu().numpy(), label='Posterior f prime')
    plt.fill_between(train_x.cpu().numpy().reshape(-1), (posterior_f_prime-std_f_prime).cpu().numpy().reshape(-1), (posterior_f_prime+std_f_prime).cpu().numpy().reshape(-1), alpha=0.5)
    plt.legend()
    plt.show()

