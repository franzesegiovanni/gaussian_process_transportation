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
from GILoSA import Transport
import pathlib
from plot_utils import plot_vector_field_minvar, plot_vector_field 
import ot
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
print("Fit Gaussian Process Dynamical System on the source distribution")
#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
plot_vector_field(gp_deltaX, x_grid,y_grid,X,S)
#%% Fit a GP to both surfaces and sample equal amount of indexed points, find delta pointcloud between old and sampled new surface
source_distribution=S
target_distribution=S1

n= source_distribution.shape[0]
a = np.ones((n,)) / n  # uniform distribution on samples

n1= target_distribution.shape[0]
b = np.ones((n1,)) / n1  # uniform distribution on samples

# # Define the cost matrix using the Euclidean distance
M = ot.dist(source_distribution, target_distribution)#+np.abs(ot.dist(source_distribution) - ot.dist(target_distribution))
M /= M.max()
lambd = 0.5e-3 #smaller is less smooth
transport_plan = ot.smooth.smooth_ot_dual(a, b, M, lambd, reg_type='kl')

row_sums = np.sum(transport_plan, axis=1)  # Compute the sum of each row
transport_plan = transport_plan / row_sums[:, np.newaxis] 
plt.figure()
cov=transport_plan @ transport_plan.transpose()
row_sums = np.sum(cov, axis=1)  # Compute the sum of each row
cov = cov / row_sums[:, np.newaxis]
plt.imshow(cov)
plt.title('covariance matrix')
# Apply the transport plan to the source cloud to obtain the transformed cloud
transformed_distribution = np.zeros_like(source_distribution)
transformed_distribution=transport_plan @ target_distribution 

entropy=np.sum(-np.log(cov) * cov, axis=1)
entropy_treshold=4
mask=entropy<entropy_treshold
# print("Entropy")
# print(entropy)
# print("Max Corr")
# print(np.max(cov, axis=1))
# print("Mask")
# print(mask)
target_distribution=transformed_distribution
#%% Transport the dynamical system on the new surface
transport=Transport()
transport.source_distribution=source_distribution[mask] #select only the point that have low entropy
transport.target_distribution=target_distribution[mask] #select only the point that have low entropy
transport.training_traj=X
transport.training_delta=deltaX
print("Fit the Gaussian Process Transportation")
transport.fit_transportation()
transport.apply_transportation()


X1=transport.training_traj
deltaX1=transport.training_delta 
x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 100)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 100)

# Fit the Gaussian Process dynamical system 
print("Fit the Gaussian Process Dynamical System on the target distribution")  
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)
plot_vector_field(gp_deltaX1, x1_grid,y1_grid,X1,S1)
plt.show()