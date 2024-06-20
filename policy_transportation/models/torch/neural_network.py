import torch
import torch.nn as nn
import torch.optim as optim
from  torch.autograd.functional import jacobian
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm 
from torch import autograd
import random

class NeuralNetwork():
    def __init__(self, X, Y):
        self.use_cuda=False
        input_size=X.shape[1]
        output_size=Y.shape[1]
        self.nn= MLP(input_size=input_size, output_size= output_size , hidden_size=100)
        if self.use_cuda:
            self.nn=self.nn.cuda()
    def fit(self, X, Y, num_epochs=100):
        # Define the loss function and optimizer
        X=torch.from_numpy(X).float()
        Y=torch.from_numpy(Y).float()
        train_dataset = TensorDataset(X, Y)
        self.train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        optimizer = optim.Adam(self.nn.parameters(), lr=0.01, weight_decay=0.1)

    
        # Training loop
        epochs_iter = tqdm(range(num_epochs))
        for epoch in epochs_iter:
            for x_batch, y_batch in tqdm(self.train_loader, desc="Minibatch", leave=False):
                # Forward pass
                if self.use_cuda:
                    x_batch=x_batch.cuda()
                    y_batch=y_batch.cuda()
                outputs = self.nn(x_batch)
                loss = criterion(outputs, y_batch)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def mean_fun(self,x):
        predictions = self.nn(x)
        predictions = torch.sum(predictions, dim=0)
        return predictions
    
    def predict(self,x):
        x=torch.from_numpy(x).float()
        if self.use_cuda:
            x=x.cuda()
        predictions = self.nn(x)
        return predictions.detach().cpu().numpy()
         
    def derivative(self, x): 
        x=torch.from_numpy(x).float()
        if self.use_cuda:
            x=x.cuda()
        J=jacobian(self.mean_fun, x).cpu().detach().numpy()
        J=J.transpose(1,0,2)    
        return J
    
# Define the MLP class
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10000):
        super(MLP, self).__init__()

        seed = random.randint(0, 2**32 - 1)
        print("Seed:", seed)
        # Set the random seed for reproducibility
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        return self.fc4(x)
        