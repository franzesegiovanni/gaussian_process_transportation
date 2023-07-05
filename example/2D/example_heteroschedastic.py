import numpy as np
import matplotlib.pyplot as plt
from GILoSA import HeteroschedasticGaussianProcess as HGPR
from GILoSA import GaussianProcess as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
# Set the random seed for reproducibility
np.random.seed(42)

# Number of data points
n = 100

# Generate the x-values
x = np.linspace(0, 10, n)

# Generate the true underlying function
y_true = 2 * x

# Generate the noise with varying variance
variance = np.abs(2 * x + 1)/4  # Varying variance based on x
noise = np.random.normal(0, variance)

# Generate the observed y-values
y_observed = y_true + noise

k = C(constant_value=np.sqrt(0.1))  * RBF (1*np.ones(1), [1,100]) + WhiteKernel(0.01) #this kernel works much better!    
gp=HGPR(kernel=k)
# gp=GPR(kernel=k)
# print(sigma_noise)
gp.fit(x.reshape(-1,1), y_observed.reshape(-1,1), sigma_noise=variance)
# gp.fit(x.reshape(-1,1), y_observed.reshape(-1,1))

mi, covar= gp.predict(x.reshape(-1,1), return_var=True, sigma_noise=variance)
# mi, std= gp.predict(x.reshape(-1,1))
# print(var.shape)
var=np.diag(covar)
std=np.sqrt(var)
print(std.shape)
# std=std.reshape(-1,1)
print(std.shape)
# print(mi.shape)
# Plot the data
plt.scatter(x, y_observed, label='Observed Data')
plt.plot(x, y_true, color='red', label='True Function')
plt.plot(x, mi, color='black', label='True Function')
y_up=mi.reshape(-1,1) + 2*std.reshape(-1,1)
y_down=mi.reshape(-1,1) - 2*std.reshape(-1,1)
print(y_up.shape)
plt.fill_between(x, y_up.reshape(-1,), y_down.reshape(-1,), alpha=0.2, color='black', label='Uncertainty')
plt.plot(x, y_up, color='black')
plt.plot(x, y_down, color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
