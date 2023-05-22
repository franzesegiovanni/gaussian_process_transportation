"""
Authors: Ravi Prakash and Giovanni Franzese, Dec 2022
Email: r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from regressor import GPR
import pathlib
from sklearn.decomposition import PCA
# %matplotlib inline

#%% Load the drawings

data =np.load(str(pathlib.Path().resolve())+'/2D/data/'+str('circletest1')+'.npz')
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
# S=S[:,:]
# S1=S1[::5,:]


#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
# %matplotlib inline
# gp_deltaX.plot_vector_field(x_grid,y_grid,X,S)
# gp_deltaX.plot_vector_field_minvar(x_grid,y_grid,X,S)


#%% Fit a GP to both surfaces and sample equal amount of indexed points, find delta pointcloud between old and sampled new surface
indexS = np.linspace(0, 1, len(S[:,0]))
indexS1 = np.linspace(0, 1, len(S1[:,0]))


k_S = C(constant_value=np.sqrt(0.1))  * RBF(0.01*np.ones(1)) + WhiteKernel(0.0001 )  
gp_S=GPR(kernel=k_S)
gp_S.fit(indexS.reshape(-1,1),S)

k_S1 = C(constant_value=np.sqrt(0.1))  * RBF(0.01*np.ones(1)) + WhiteKernel(0.0001 )  
gp_S1=GPR(kernel=k_S1)
gp_S1.fit(indexS1.reshape(-1,1),S1)

index = np.linspace(0, 1, 100).reshape(-1,1)
deltaPC = np.zeros((len(index),2))

S_sampled, _ =gp_S.predict(index)   
S1_sampled, _ =gp_S1.predict(index)

# S1_sampled= np.flip(S1_sampled,0)
# Calculate the orientation and center of mass of the point-cloud using starting and ending point 

#This method is not general and will not scale with 3D data. However, using the PCA does not return a rotation matrix and cannot be use either. 
# The orientation of the surface should be known from an optitrack system or similar. 

S_sampled_mean=np.mean(S_sampled,0)
S1_sampled_mean=np.mean(S1_sampled,0)

S_sampled=S_sampled-S_sampled_mean

S1_sampled=S1_sampled-S1_sampled_mean

R0_=S_sampled[-1]-S_sampled[0]
R0_=R0_/np.linalg.norm(R0_)

R1_=S1_sampled[-1]-S1_sampled[0]
R1_=R1_/np.linalg.norm(R1_)

R0=np.zeros((2,2))

R0[0,0]=R0_[0]
R0[1,0]=R0_[1]
R0[0,1]=-R0_[1]
R0[1,1]=R0_[0]

R1=np.zeros((2,2))

R1[0,0]=R1_[0]
R1[1,0]=R1_[1]
R1[0,1]=-R1_[1]
R1[1,1]=R1_[0]

# Let's transform our surface in their principal components

S_sampled=S_sampled @ R0

S1_sampled=S1_sampled @ R1


deltaPC = S1_sampled - S_sampled

# mean_deltaPC =  np.mean(deltaPC,axis=0)
# print(mean_deltaPC)
deltaPC = deltaPC
# fig = plt.figure(figsize = (12, 7))
# plt.xlim([-50, 50-1])
# plt.ylim([-50, 50-1])
# plt.scatter(S1_sampled[:,0],S1_sampled[:,1], color=[1,0,0]) 
# plt.scatter(S_sampled[:,0],S_sampled[:,1], color=[0,1,0])   
# plt.scatter(S[:,0],S[:,1], color=[0,0,0])   
# plt.scatter(deltaPC[:,0],deltaPC[:,1]) 



#%% Fit a GP to delta pointcloud and Find demo for the new surface
k_deltaPC = C(constant_value=np.sqrt(10))  * Matern(1*np.ones(2), [5,10], nu=0.5) + WhiteKernel(0.0001 )
# k_deltaPC = C(constant_value=np.sqrt(0.1)) * RBF(1*np.ones(2), [1,10]) + WhiteKernel(0.01 )
# k_deltaPC = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(2), [5,20]) + WhiteKernel(0.01 )

gp_deltaPC = GPR(kernel=k_deltaPC)
gp_deltaPC.fit(S_sampled,deltaPC)

# deltaPCX = np.zeros((len(X[:,0]),2))
# for i in range(len(X[:,0])):
#     pos=np.array(X[i,:]).reshape(1,-1)
#     [y_S1, std_S1]=gp_deltaPC.predict(pos)
#     deltaPCX[i,0]= y_S1[0][0]
#     deltaPCX[i,1]= y_S1[0][1]

# X_pca= pca.transform(X)

X_pca= (X-S_sampled_mean) @ R0

deltaPCX, _= gp_deltaPC.predict(X_pca)
print("Lenghtscale of the deltaPCX", gp_deltaPC.kernel_params_[0])
# To create the mapping we need to deform, rotate and then traslate. 
X1= (X_pca+deltaPCX) @ np.transpose(R1) + S1_sampled_mean 
fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X1[:,0],X1[:,1], color=[1,0,0]) 
plt.scatter(S1[:,0],S1[:,1], color=[0,1,0])   
plt.scatter(X[:,0],X[:,1], color=[0,0,1]) 
plt.scatter(S[:,0],S[:,1], color=[0,0,0])   
# plt.scatter(S_sampled[:,0],deltaPC[:,1]) 
plt.show()


x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 100)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 100)

#%% Fit a dynamical system for the new demo (Method 1)
# print("Method that recreates delta from the data translation")
# deltaX1=np.zeros((len(X1),2))
# for j in range(len(X1)-1): 
#     deltaX1[j,:]=(X1[j+1,:]-X1[j,:])

# k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
# gp_deltaX1=GPR(kernel=k_deltaX1)
# gp_deltaX1.fit(X1, deltaX1)
# # gp_deltaX1.plot_vector_field(x1_grid,y1_grid,X1,S1)
# gp_deltaX1.plot_vector_field_minvar(x1_grid,y1_grid,X1,S1)






#%%  Fit a dynamical system for the new demo (Method 2)
print("Methos using the Jacobian")
deltaX1=np.ones((len(X),2))
for i in range(len(X[:,0])):
    pos=(np.array(X[i,:]).reshape(1,-1) - S_sampled_mean) @ R0
    [Jacobian,_]=gp_deltaPC.derivative(pos)
    #Be carefull. You need to compute J^T*Delta_x. (J= ddeltaPC_dX 
    # deltaX1[i,0]=deltaX[i,0]+(Jacobian[0][0][0])*(deltaX[i,0]) + (Jacobian[0][1][0]*deltaX[i,1])
    # deltaX1[i,1]=deltaX[i,1]+(Jacobian[0][0][1]*deltaX[i,0]) + (Jacobian[0][1][1])*(deltaX[i,1])
    deltaX1[i]=(deltaX[i]+np.matmul(np.transpose(Jacobian[0]),deltaX[i])) @ np.transpose(R1)
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)
# gp_deltaX1.plot_vector_field(x_grid,y_grid,X1,S1)
gp_deltaX1.plot_vector_field_minvar(x1_grid,y1_grid,X1,S1)

#%% Fit a dynamical system for a new surface (Method 3)
# print("This method uses the delta but without linearizing")
# deltaX1=np.ones((len(X),2))
# for i in range(len(X[:,0])):
#     pos=(np.array(X[i,:]).reshape(1,-1) - S_sampled_mean ) @ R0
#     delta=np.array(deltaX[i,:]).reshape(1,-1) @ R0
#     [PC_pos,_]=gp_deltaPC.predict(pos)
#     [PC_pos_plus_delta,_]=gp_deltaPC.predict(pos+delta)
#     # deltaX1[i,:]=deltaX[i,:]+PC_pos_plus_delta-PC_pos
#     deltaX1[i,:]=(delta+PC_pos_plus_delta-PC_pos) @ np.transpose(R1)
#     # deltaX1[i,0]=(1+ddeltaPC_dX[0][0][0])*(deltaX[i,0]) + (ddeltaPC_dX[0][0][1]*deltaX[i,1])
#     # deltaX1[i,1]=(ddeltaPC_dX[0][1][0]*deltaX[i,0]) + (1+ddeltaPC_dX[0][1][1])*(deltaX[i,1])

# k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
# gp_deltaX1=GPR(kernel=k_deltaX1)
# gp_deltaX1.fit(X1, deltaX1)
# # gp_deltaX1.plot_vector_field(x_grid,y_grid,X1,S1)
# gp_deltaX1.plot_vector_field_minvar(x1_grid,y1_grid,X1,S1)
