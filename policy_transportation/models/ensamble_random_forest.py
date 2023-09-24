
from sklearn.ensemble import RandomForestRegressor
import random 
import numpy as np

class Ensemble_RF():
    def __init__(self, n_estimators, max_depth):
        self.n_estimators = n_estimators
        self.ensemble = [RandomForestRegressor(random_state=random.randint(0, 2**32 - 1), max_depth=max_depth) for _ in range(n_estimators)]

    def fit(self, X, y):
        # Train the ensemble of random forests
        for rf in self.ensemble:
            rf.fit(X, y)

    def predict(self, X, return_std=False):
        predictions = [rf.predict(X) for rf in self.ensemble]
        predictions = np.array(predictions)  # Shape: (n_estimators, n_samples)

        # Calculate the mean and variance of the predictions
        mean = np.mean(predictions, axis=0)
        if return_std:
            variance_predictions = np.var(predictions, axis=0)
            std = np.sqrt(variance_predictions)
            return mean, std
        return mean
    
    def samples(self, X):
        predictions = [nn.predict(X) for nn in self.ensemble]
        predictions = np.array(predictions)  # Shape: (n_estimators, n_samples)
        return predictions