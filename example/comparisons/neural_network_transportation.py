"""
Authors:  Giovanni Franzese
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from GILoSA import GaussianProcess as GPR 
from GILoSA import Transport
import pathlib
from plot_utils import plot_vector_field_minvar, plot_vector_field , draw_error_band
import warnings
from GILoSA import AffineTransform
import random
from models import Ensamble_NN
warnings.filterwarnings("ignore")
#%% Load the drawings

data =np.load(str(pathlib.Path().resolve())+'/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(S[:,0],S[:,1], color=[0,1,0])   
plt.scatter(S1[:,0],S1[:,1], color=[0,0,1]) 
plt.legend(["Demonstration","Surface","New Surface"])


#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])

## Downsample
X=X[::2,:]
deltaX=deltaX[::2,:]

#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
plot_vector_field(gp_deltaX, x_grid,y_grid,X,S)

#%% Fit a GP to both surfaces and sample equal amount of indexed points, find delta pointcloud between old and sampled new surface
indexS = np.linspace(0, 1, len(S[:,0]))
indexS1 = np.linspace(0, 1, len(S1[:,0]))
k_S = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(1)) + WhiteKernel(0.01 )  
gp_S=GPR(kernel=k_S)
gp_S.fit(indexS.reshape(-1,1),S)

k_S1 = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(1)) + WhiteKernel(0.01 )  
gp_S1=GPR(kernel=k_S1)
gp_S1.fit(indexS1.reshape(-1,1),S1)

index = np.linspace(0, 1, 100).reshape(-1,1)
deltaPC = np.zeros((len(index),2))

source_distribution, _  =gp_S.predict(index)   
target_distribution, _  =gp_S1.predict(index)

affine_transform=AffineTransform()
affine_transform.fit(source_distribution, target_distribution)
source_distribution=affine_transform.predict(source_distribution) 
delta_distribution = target_distribution - source_distribution
# model = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100, 100, 50, 20), max_iter=10000, random_state=random.randint(0, 2**32 - 1))
model = Ensamble_NN(n_estimators=50, hidden_layer_sizes=(100, 100, 100, 100, 100, 50, 20))
model.fit(source_distribution, delta_distribution)

# Make predictions on the test set
X1=affine_transform.predict(X)
X1_gp, std = model.predict(X1, return_std=True)
# print(std)
std= np.sqrt(std[:,0]**2+std[:,1]**2)
# print(std)
X1_gp=X1+X1_gp
X_samples= X1+model.samples(X1)
print(X_samples.shape)
fig, ax = plt.subplots()
ax.scatter(X1_gp[:,0],X1_gp[:,1], cmap='rainbow')
ax.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,0])
ax.plot(X_samples[:,:,0].T, X_samples[:,:,1].T)
draw_error_band(ax, X1_gp[:,0], X1_gp[:,1], err=2*std[:], facecolor= [255.0/256.0,140.0/256.0,0.0], edgecolor="none", alpha=.7)
plt.show()
