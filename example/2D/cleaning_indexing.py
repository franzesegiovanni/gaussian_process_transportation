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
# %matplotlib inline

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
gp_deltaX.plot_vector_field_minvar(x_grid,y_grid,X,S)


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

# S_sampled = np.zeros((len(index),2))
# S1_sampled= np.zeros((len(index),2))

# for i in range(len(index)):
#     pos=np.array(index[i]).reshape(1,-1)
#     [xS_sampled, stdxS_sampled]=gp_Sx.predict(pos)
#     [yS_sampled, stdyS_sampled]=gp_Sy.predict(pos)
#     [xS1_sampled, stdxS1_sampled]=gp_S1x.predict(pos)
#     [yS1_sampled, stdyS1_sampled]=gp_S1y.predict(pos)

 
#     S_sampled[i,:]= (xS_sampled,yS_sampled)
#     S1_sampled[i,:]= (xS1_sampled,yS1_sampled)

S_sampled, _ =gp_S.predict(index)   
S1_sampled, _ =gp_S1.predict(index)

deltaPC = S1_sampled - S_sampled

mean_deltaPC =  np.mean(deltaPC,axis=0)
print(mean_deltaPC)
deltaPC = deltaPC - mean_deltaPC
fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(S1_sampled[:,0],S1_sampled[:,1], color=[1,0,0]) 
plt.scatter(S_sampled[:,0],S_sampled[:,1], color=[0,1,0])   
# plt.scatter(S[:,0],S[:,1], color=[0,0,0])   
# plt.scatter(deltaPC[:,0],deltaPC[:,1]) 



#%% Fit a GP to delta pointcloud and Find demo for the new surface
k_deltaPC = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(2)) + WhiteKernel(0.01 )
# k_deltaPC = C(constant_value=np.sqrt(0.1)) * RBF(1*np.ones(2), [1,10]) + WhiteKernel(0.01 )
# k_deltaPC = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(2), [5,20]) + WhiteKernel(0.01 )

gp_deltaPC = GPR(kernel=k_deltaPC)
gp_deltaPC.fit(S_sampled,deltaPC)

deltaPCX = np.zeros((len(X[:,0]),2))
for i in range(len(X[:,0])):
    pos=np.array(X[i,:]).reshape(1,-1)
    [y_S1, std_S1]=gp_deltaPC.predict(pos)
    deltaPCX[i,0]= y_S1[0][0]
    deltaPCX[i,1]= y_S1[0][1]

X1 = X+deltaPCX+mean_deltaPC
fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X1[:,0],X1[:,1], color=[1,0,0]) 
plt.scatter(S1[:,0],S1[:,1], color=[0,1,0])   
plt.scatter(X[:,0],X[:,1], color=[0,0,1]) 
plt.scatter(S[:,0],S[:,1], color=[0,0,0])   
# plt.scatter(S_sampled[:,0],deltaPC[:,1]) 

# #%% Fit a dynamical system for the new demo (Method 1)
print("This is the first method")
deltaX1=np.zeros((len(X1),2))
for j in range(len(X1)-1): 
    deltaX1[j,:]=(X1[j+1,:]-X1[j,:])

k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)
x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 100)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 100)
# gp_deltaX1.plot_vector_field(x1_grid,y1_grid,X1,S1)
gp_deltaX1.plot_vector_field_minvar(x1_grid,y1_grid,X1,S1)



# #%%  Fit a dynamical system for the new demo (Method 2)
# print("This is the second method")
# deltaX1=np.ones((len(X),2))
# for i in range(len(X[:,0])):
#     pos=np.array(X[i,:]).reshape(1,-1)
#     [Jacobian,_]=gp_deltaPC.derivative(pos)
#     #Be carefull. You need to compute J^T*Delta_x. (J= ddeltaPC_dX 
#     # deltaX1[i,0]=deltaX[i,0]+(Jacobian[0][0][0])*(deltaX[i,0]) + (Jacobian[0][1][0]*deltaX[i,1])
#     # deltaX1[i,1]=deltaX[i,1]+(Jacobian[0][0][1]*deltaX[i,0]) + (Jacobian[0][1][1])*(deltaX[i,1])
#     deltaX1[i]=deltaX[i]+np.matmul(np.transpose(Jacobian[0]),deltaX[i])
# k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
# gp_deltaX1=GPR(kernel=k_deltaX1)
# gp_deltaX1.fit(X1, deltaX1)
# # gp_deltaX1.plot_vector_field(x_grid,y_grid,X1,S1)
# gp_deltaX1.plot_vector_field_minvar(x1_grid,y1_grid,X1,S1)

# #%% Fit a dynamical system for a new surface (Method 3)
# print("This is the third method")
# deltaX1=np.ones((len(X),2))
# for i in range(len(X[:,0])):
#     pos=np.array(X[i,:]).reshape(1,-1)
#     delta=np.array(deltaX[i,:]).reshape(1,-1)
#     [PC_pos,_]=gp_deltaPC.predict(pos)
#     [PC_pos_plus_delta,_]=gp_deltaPC.predict(pos+delta)
#     deltaX1[i,:]=deltaX[i,:]+PC_pos_plus_delta-PC_pos
#     # deltaX1[i,0]=(1+ddeltaPC_dX[0][0][0])*(deltaX[i,0]) + (ddeltaPC_dX[0][0][1]*deltaX[i,1])
#     # deltaX1[i,1]=(ddeltaPC_dX[0][1][0]*deltaX[i,0]) + (1+ddeltaPC_dX[0][1][1])*(deltaX[i,1])

# k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
# gp_deltaX1=GPR(kernel=k_deltaX1)
# gp_deltaX1.fit(X1, deltaX1)
# # gp_deltaX1.plot_vector_field(x_grid,y_grid,X1,S1)
# gp_deltaX1.plot_vector_field_minvar(x1_grid,y1_grid,X1,S1)
