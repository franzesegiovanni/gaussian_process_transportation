"""
Authors:  Giovanni Franzese and Ravi Prakash, Dec 2022
Email: g.franzese@tudelft.nl, r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from GILoSA import GaussianProcess as GPR
from GILoSA import HeteroschedasticGaussianProcess as HGPR
from GILoSA import Transport
import pathlib
from plot_utils import plot_vector_field_minvar, plot_vector_field 
import warnings
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
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01) 
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

#%% Transport the dynamical system on the new surface
transport=Transport()
transport.source_distribution=source_distribution 
transport.target_distribution=target_distribution
transport.training_traj=X
transport.training_delta=deltaX
print('Transporting the dynamical system on the new surface')
k_transport = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(2)) + WhiteKernel(0.01 )
transport.kernel_transport=k_transport
transport.fit_transportation()
transport.apply_transportation()
X1=transport.training_traj
deltaX1=transport.training_delta 
x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 100)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 100)

sigma_noise=[]
# Calculate the noise on the transported trajectory using the epistemic uncertainty of the GP
for i in range(len(X)):
    [Jacobian,_]=transport.gp_delta_map.derivative(X[i].reshape(1,-1))
    _, std= transport.gp_delta_map.predict(X[i].reshape(1,-1))
    var_derivative=std**2/(np.linalg.norm(transport.gp_delta_map.kernel_params_[0]))**2
    # compute the determinant of the Jacobian
    # det_Jacobian = np.linalg.det(Jacobian[0])
    # scale the standard deviation by the determinant of the Jacobian
    sigma_noise.append(var_derivative)


sigma_noise=np.array(sigma_noise)
# print(sigma_noise)
# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX1=HGPR(kernel=k_deltaX1)
# print(sigma_noise)
gp_deltaX1.fit(X1, deltaX1, sigma_noise=sigma_noise)
plot_vector_field(gp_deltaX1, x1_grid,y1_grid,X1,S1)
# plt.show()


X, Y = np.meshgrid(x1_grid, y1_grid)
input=np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))

sigma_noise_prediction=[]
for i in range(len(input)):
    [Jacobian,_]=transport.gp_delta_map.derivative(input[i].reshape(1,-1))
    _, std= transport.gp_delta_map.predict(input[i].reshape(1,-1))
    var_derivative=std**2/(np.linalg.norm(transport.gp_delta_map.kernel_params_[0]))**2
    # compute the determinant of the Jacobian
    # det_Jacobian = np.linalg.det(Jacobian[0])
    # scale the standard deviation by the determinant of the Jacobian
    sigma_noise_prediction.append(var_derivative)

sigma_noise_prediction=np.array(sigma_noise_prediction)

# print(sigma_noise_prediction)
sigma_total=gp_deltaX1.predict(input, sigma_noise=sigma_noise_prediction, return_var=True)[1]

# sigma_total=np.sqrt(sigma_total)
std_total=np.sqrt(np.diag(sigma_total))
# print(std_total.shape)
# std_total.shape
std_total=std_total.reshape(X.shape)

std_aliatoric=np.sqrt(sigma_noise_prediction)
std_aliatoric=std_aliatoric.reshape(X.shape)
# ax.scatter3D(X1[:,0], X1[:,1], np.zeros(len(X1[:,0])), 'r')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, std_total, linewidth=0, antialiased=True, cmap=plt.cm.inferno)
# plt.axis('off')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, std_aliatoric, linewidth=0, antialiased=True, cmap=plt.cm.inferno)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, np.sqrt(std_aliatoric**2+std_total**2), linewidth=0, antialiased=True, cmap=plt.cm.inferno)
plt.show()