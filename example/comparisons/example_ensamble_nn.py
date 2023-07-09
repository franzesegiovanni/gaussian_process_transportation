import numpy as np
import matplotlib.pyplot as plt
from models import Ensamble_NN, Ensemble_RF
# Generate noisy sine wave data


X_train = np.linspace(0, 10, 100).reshape(-1, 1)
y_train = np.sin(X_train) + 0.2 * np.random.randn(len(X_train)).reshape(-1, 1)

methods=[Ensamble_NN(n_estimators=10), Ensemble_RF(n_estimators=10, max_depth=5)]
names=['Neural Network', 'Random Forest']

for ENN , name in zip(methods, names):

    ENN.fit(X_train, y_train)
    # Generate test data
    X_test = np.linspace(-10, 20, 1000).reshape(-1, 1)

    # Make predictions using the ensemble
    mean, std= ENN.predict(X_test, return_std=True)


    predictions= ENN.samples(X_test)
    # Plot the true function, noisy data, and ensemble predictions
    plt.figure(figsize=(10, 6))
    plt.plot(X_train, y_train, 'o', label='Noisy Data')
    plt.plot(X_test, np.sin(X_test), label='True Function', linewidth=2)
    plt.plot(X_test, mean, label='Ensemble Mean Prediction', linewidth=2)
    plt.plot(X_test, predictions.T)
    plt.fill_between(X_test.flatten(), mean - 2* std,
                     mean + 2* std,
                     alpha=0.3, label='Uncertainty (±2 std)')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Ensemble '+name)
    plt.legend()
plt.show()
