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
from GILoSA import AffineTransform
import pathlib
from plot_utils import plot_vector_field_minvar, plot_vector_field 

#%% Load the drawings

data =np.load(str(pathlib.Path().resolve())+'/data/'+str('indexing')+'.npz')
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
# plot_vector_field_minvar(gp_deltaX, x_grid,y_grid,X,S)

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

# Apply Affine transformation 
affine_transform=AffineTransform()

affine_transform.fit(source_distribution, target_distribution)

source_distribution=affine_transform.predict(source_distribution[:,:])

plt.scatter(source_distribution[:,0],source_distribution[:,1])  

X_rotated=affine_transform.predict(X[:,:])

plt.scatter(X_rotated[:,0],X_rotated[:,1]) 

plt.show()
delta_distribution = target_distribution - source_distribution
# 
# 
#%% Fit a GP to delta pointcloud and Find demo for the new surface
k_transport = C(constant_value=np.sqrt(0.1))  * RBF(np.ones(2)) + WhiteKernel(0.01)
gp_transport = GPR(kernel=k_transport)
gp_transport.fit(source_distribution,delta_distribution)
# 
X_roto_trans=affine_transform.predict(X)
[delta_transport,_]=gp_transport.predict(X_roto_trans)
# 
X1 = X_roto_trans+delta_transport
fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X1[:,0],X1[:,1], color=[1,0,0]) 
plt.scatter(S1[:,0],S1[:,1], color=[0,1,0])   
plt.scatter(X[:,0],X[:,1], color=[0,0,1]) 
plt.scatter(S[:,0],S[:,1], color=[0,0,0])   
x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 100)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 100)
# 
# 
#%%  Fit a dynamical system for the new demo (Method 2)
deltaX1=np.ones((len(X),2))
for i in range(len(X[:,0])):
    pos=np.array(X[i,:]).reshape(1,-1)
    pos_rot_trans=affine_transform.predict(pos)
    [Jacobian,_]=gp_transport.derivative(pos_rot_trans)
    R=(np.eye(2)+np.transpose(Jacobian[0])) @ affine_transform.rotation_matrix
# 
    deltaX1[i]= R @ deltaX[i]
# 
# Fit the Gaussian Process again    
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)
# gp_deltaX1.plot_vector_field(x_grid,y_grid,X1,S1)
plot_vector_field_minvar(gp_deltaX1, x1_grid,y1_grid,X1,S1)
# 
# 