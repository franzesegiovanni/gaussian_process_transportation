from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib import pyplot as plt

X=np.linspace(0, 1, 100).reshape(-1,1)
Y=np.sin(2*np.pi*X)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = X_train#scaler.fit_transform(X_train)
X_test_scaled = X_test#scaler.transform(X_test)

# Create and train the MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100, 50, 50, 50, 50, 50), max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
# Y1 = model.predict(scaler.transform(X))
Y1 = model.predict(X)


plt.plot(X,Y1)
plt.scatter(X,Y, color=[1,0,0], alpha=0.1)
plt.show()
