from policy_transportation.models.torch.neural_network import NeuralNetwork
import torch 
import numpy as np

class EnsembleNeuralNetwork():
    def __init__ (self, X,Y, n_estimators=10):
        self.ensemble = [NeuralNetwork(X,Y) for _ in range(n_estimators)]
    
    def fit(self, X, Y, num_epochs=100):
        # Train the ensemble of neural networks
        index=1
        for nn in self.ensemble:
            print("Training a neural network number:", index, "out of", len(self.ensemble))
            nn.fit(X, Y, num_epochs)
            index+=1

    def predict(self, X, return_std=False):
        predictions = [nn.predict(X) for nn in self.ensemble]
        predictions = np.array(predictions)  # Shape: (n_estimators, n_samples)

        # Calculate the mean and variance of the predictions
        mean = np.mean(predictions, axis=0)
        if return_std:
            # variance_predictions = np.var(predictions, axis=0)
            # print("Predection shape", predictions.shape)
            std= np.std(predictions, axis=0)
            # print(std)
            return mean, std
        return mean
    
    def derivative(self, x, return_std=False): 
        predictions = [nn.derivative(x) for nn in self.ensemble]
        predictions = np.array(predictions)  # Shape: (n_estimators, n_samples)

        # Calculate the mean and variance of the predictions
        mean = np.mean(predictions, axis=0)
        if return_std:
            # variance_predictions = np.var(predictions, axis=0)
            # print("Predection shape", predictions.shape)
            std= np.std(predictions, axis=0)
            # print(std)
            return mean, std
        return mean
    
    def samples(self, X):
        predictions = [nn.predict(X) for nn in self.ensemble]
        predictions = np.array(predictions)  # Shape: (n_estimators, n_samples)
        return predictions