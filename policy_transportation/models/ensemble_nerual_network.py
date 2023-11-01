from sklearn.neural_network import MLPRegressor
import random 
import numpy as np
class Ensemble_NN():
    def __init__ (self, n_estimators, hidden_layer_sizes=(100, 100, 100,100)):
        self.ensemble = [MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random.randint(0, 2**32 - 1), alpha=1, solver='adam') for _ in range(n_estimators)]
    
    def fit(self, X, y):
        # Train the ensemble of neural networks
        for nn in self.ensemble:
            nn.fit(X, y)

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
    
    def samples(self, X):
        predictions = [nn.predict(X) for nn in self.ensemble]
        predictions = np.array(predictions)  # Shape: (n_estimators, n_samples)
        return predictions