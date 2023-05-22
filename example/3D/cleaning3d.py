"""
Authors: Ravi Prakash and Giovanni Franzese, Dec 2022
Email: r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""


import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from regressor3d import GPR
import pathlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



# Load the drawings

data =np.load(str(pathlib.Path().resolve())+'/data/'+str('last_spiral')+'.npz')
X=data['demo'] 
S=data['old_surface'] 
S1=data['new_surface']


fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlim(min(X[:,0]), max(X[:,0]))
ax.set_ylim(min(X[:,1]), max(X[:,1]))
ax.set_zlim(min(X[:,2]), max(X[:,2]))
newsurf = ax.plot_surface(S1[:,:,0], S1[:,:,1], S1[:,:,2], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf = ax.plot_surface(S[:,:,0], S[:,:,1], S[:,:,2], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(X[:,0], X[:,1],X[:,2], color='red') # parabola points
plt.show()


# Calculate deltaX
deltaX = np.zeros((len(X),3))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])



# k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(3), (5,10), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(3),  nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)


x_grid=np.linspace(np.min(X[:,0]), np.max(X[:,0]), 10)
y_grid=np.linspace(np.min(X[:,1]), np.max(X[:,1]), 10)
z_grid=np.linspace(np.min(X[:,2]), np.max(X[:,2]), 3)
gp_deltaX.plot_traj_evolution(x_grid,y_grid,z_grid,X,S)


S1_ = S1.reshape(-1,3)
S_ = S.reshape(-1,3)

deltaPC = S1_ - S_




# # Downsample
S1_=S1_[::3]
S_=S_[::3]
deltaPC=deltaPC[::3]





# Fit a GP to delta pointcloud and Find demo for the new surface
k_deltaPC = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(3)) + WhiteKernel(0.01 )
gp_deltaPC = GPR(kernel=k_deltaPC)
gp_deltaPC.fit(S_,deltaPC)

deltaPCX = np.zeros((len(X[:,0]),3))
for i in range(len(X[:,0])):
    pos=np.array(X[i,:]).reshape(1,-1)
    [y_S1, std_S1]=gp_deltaPC.predict(pos)
    deltaPCX[i,0]= y_S1[0][0]
    deltaPCX[i,1]= y_S1[0][1]
    deltaPCX[i,2]= y_S1[0][2]

X1 = X+deltaPCX

# Fit a dynamical system for the new demo (Method 1)
print("This is the first method")
deltaX1=np.zeros((len(X1),3))
for j in range(len(X1)-1): 
    deltaX1[j,:]=(X1[j+1,:]-X1[j,:])

k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(3), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)
gp_deltaX1.plot_traj_evolution(x_grid,y_grid,z_grid,X1,S1)