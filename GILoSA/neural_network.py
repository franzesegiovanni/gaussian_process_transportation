import torch
import torch.nn as nn
import torch.optim as optim
from  torch.autograd.functional import jacobian
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm 

# Define the MLP class
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100):
        super(MLP, self).__init__()
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


class NeuralNetwork():
    def __init__(self, X, Y):
        # self.nn= SVGP_LMC(num_latents=1, X=X, Y=Y ,num_task=Y.shape[1], num_inducing=num_inducing)
        input_size=X.shape[1]
        output_size=Y.shape[1]
        self.nn= MLP(input_size=input_size, output_size= output_size , hidden_size=100)
        if torch.cuda.is_available():
            self.nn=self.nn.cuda()
        X=torch.from_numpy(X).float()
        Y=torch.from_numpy(Y).float()
        train_dataset = TensorDataset(X, Y)
        self.train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True) 
    def fit(self, num_epochs=100):
        # Define the loss function and optimizer

        criterion = nn.MSELoss()  # Mean Squared Error for regression
        optimizer = optim.Adam(self.nn.parameters(), lr=0.01, weight_decay=0.1)

    
        # Training loop
        epochs_iter = tqdm(range(num_epochs))
        for epoch in epochs_iter:
            for x_batch, y_batch in tqdm(self.train_loader, desc="Minibatch", leave=False):
                # Forward pass
                if torch.cuda.is_available():
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
        return predictions
    
    def predict(self,x):
        x=torch.from_numpy(x).float()
        if torch.cuda.is_available():
            x=x.cuda()
        predictions = self.nn(x)
        return predictions.detach().cpu().numpy()
         
    def derivative(self, x): 
        x=torch.from_numpy(x).float()
        if torch.cuda.is_available():
            x=x.cuda()
        return jacobian(self.mean_fun, x).cpu().detach().numpy()
    
