"""
Authors:  Giovanni Franzese 
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from policy_transportation import GaussianProcess as GPR
from policy_transportation import GaussianProcessTransportation as Transport
import pathlib
from policy_transportation.utils import resample
import warnings
from policy_transportation.plot_utils import draw_error_band
warnings.filterwarnings("ignore")
#%% Load the drawings
def create_vectorfield(model,datax_grid,datay_grid):
    dataXX, dataYY = np.meshgrid(datax_grid, datay_grid)
    pos = np.column_stack((dataXX.ravel(), dataYY.ravel()))
    vel, std = model.predict(pos)
    u, v = vel[:, 0].reshape(dataXX.shape), vel[:, 1].reshape(dataXX.shape)
    return u, v, std

source_path = str(pathlib.Path(__file__).parent.absolute())  
data =np.load(source_path+ '/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
X=resample(X, num_points=100)
source_distribution=resample(S,num_points=20)
target_distribution=resample(S1, num_points=20)

#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])


fig, axs = plt.subplots(nrows=2, ncols=2)
axs[0, 0].grid(True, alpha=0.2)
axs[0, 1].grid(True, alpha=0.2)
axs[1, 0].grid(True, alpha=0.2)
axs[1, 1].grid(True, alpha=0.2)
# Adjust the spacing between subplots
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.0) 
axs[0, 0].set_xticklabels([])
axs[0, 0].tick_params(axis='both', length=0)
axs[0, 0].set_yticklabels([])

axs[0, 1].set_xticklabels([])
axs[0, 1].tick_params(axis='both', length=0)
axs[0, 1].set_yticklabels([])

axs[1, 0].set_xticklabels([])
axs[1, 0].tick_params(axis='both', length=0)
axs[1, 0].set_yticklabels([])

axs[1, 1].set_xticklabels([])
axs[1, 1].tick_params(axis='both', length=0)
axs[1, 1].set_yticklabels([])



#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_lim=[np.min(X[:,0]-10), np.max(X[:,0]+20)]
y_lim=[np.min(X[:,1]-5), np.max(X[:,1]+30)]
x_grid=np.linspace(x_lim[0], x_lim[1], 100)
y_grid=np.linspace(y_lim[0], y_lim[1], 100)


axs[0, 0].scatter(X[:,0],X[:,1], color=[1,0,0])
axs[0, 0].scatter(source_distribution[:,0],source_distribution[:,1], color=[0,1,0])
axs[0, 0].set_xlim(x_lim)
axs[0, 0].set_ylim(y_lim)

u,v, std=create_vectorfield(gp_deltaX, x_grid,y_grid)
var=np.sum(std**2,axis=1)
std=np.sqrt(var) 
std=std.reshape(u.shape)

axs[0, 1].streamplot(x_grid, y_grid, u, v, density = 1, color=std, cmap='plasma')

axs[0, 1].scatter(X[:,0],X[:,1], color=[1,0,0])
axs[0, 1].scatter(source_distribution[:,0],source_distribution[:,1],color=[0,1,0])
axs[0, 1].set_xticklabels([])
axs[0, 1].set_yticklabels([])


axs[1, 0].set_xticklabels([])
axs[1, 0].set_yticklabels([])
axs[1, 1].set_xticklabels([])
axs[1, 1].set_yticklabels([])

#%% Transport the dynamical system on the new surface
transport=Transport()
transport.source_distribution=source_distribution 
transport.target_distribution=target_distribution
transport.training_traj=X
transport.training_delta=deltaX


k_transport = C(constant_value=np.sqrt(0.1))  * RBF(40*np.ones(2), length_scale_bounds=[0.01, 500]) + WhiteKernel(0.01, noise_level_bounds=[0.01, 0.1] )
transport.kernel_transport=k_transport
print('Transporting the dynamical system on the new surface')
transport.fit_transportation()
transport.apply_transportation()
X1=transport.training_traj
deltaX1=transport.training_delta 
std=transport.std

axs[1, 0].scatter(X1[:,0],X1[:,1], color=[1,0,0])
axs[1, 0].scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1])
axs[1, 0].set_xlim(x_lim)
axs[1, 0].set_ylim(y_lim)
axs[1, 0].set_xticklabels([])
axs[1, 0].set_yticklabels([])
draw_error_band(axs[1, 0], X1[:,0], X1[:,1], err=2*std[:], facecolor= [255.0/256.0,140.0/256.0,0.0], edgecolor="none", alpha=.4, loop=True)


#We should fit a gp for the aleatoric noise. 
kernel_uncertainty=C(constant_value=np.sqrt(0.1))  * RBF(4*np.ones(2), length_scale_bounds=[0.01, 500]) + WhiteKernel(0.01, noise_level_bounds=[0.01, 0.1] )
GP_aleatoric=GPR(kernel=kernel_uncertainty)
GP_aleatoric.fit(X1, std/transport.gp_delta_map.kernel_params_[0])

dataXX, dataYY = np.meshgrid(x_grid, y_grid)
pos = np.column_stack((dataXX.ravel(), dataYY.ravel()))
std_aleatoric, _ = GP_aleatoric.predict(pos)

var_aleatoric=std_aleatoric**2

std_aleatoric=np.sqrt(var_aleatoric)

print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01 )    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)

u,v, std_epi=create_vectorfield(gp_deltaX1, x_grid,y_grid)

var_epi= std_epi**2
var_hetero= var_epi+ var_aleatoric
std_hetero=np.sqrt(np.sum(var_hetero,1))

std_hetero=std_hetero.reshape(u.shape)
axs[1, 1].streamplot(dataXX, dataYY, u, v, density = 1, color=std_hetero, cmap='plasma')
axs[1, 1].scatter(X1[:,0],X1[:,1], color=[1,0,0])
axs[1, 1].scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1])
axs[1, 1].set_xticklabels([])
axs[1, 1].set_yticklabels([])

# Plot surface of the norm of ouput uncertainties
fig=plt.figure()
ax = fig.add_subplot(131, projection='3d')
ax.set_title('Aleatoric Uncertainty')
surf = ax.plot_surface(dataXX, dataYY, np.sum(std_aleatoric,1).reshape(u.shape) , linewidth=0, antialiased=True, cmap=plt.cm.inferno)
Z=np.sum(GP_aleatoric.Y,1)
plt.plot(X1[:,0],X1[:,1], Z , 'o', color='black', markersize=2)

ax.set_zlim(np.max(std_hetero), np.min(Z))
ax.set_ylim(np.max(dataYY), np.min(dataYY))
ax = fig.add_subplot(132, projection='3d')
ax.set_title('Episthemic Uncertainty')
Z=np.sum(std_epi,1).reshape(u.shape)
surf = ax.plot_surface(dataXX, dataYY, Z , linewidth=0, antialiased=True, cmap=plt.cm.inferno)
ax.set_zlim(np.max(std_hetero), np.min(Z))
ax.set_ylim(np.max(dataYY), np.min(dataYY))
ax = fig.add_subplot(133, projection='3d')
ax.set_title('Heteroschedastic Uncertainty')
surf = ax.plot_surface(dataXX, dataYY, std_hetero , linewidth=0, antialiased=True, cmap=plt.cm.inferno)
ax.set_zlim(np.max(std_hetero), np.min(std_hetero))
ax.set_ylim(np.max(dataYY), np.min(dataYY))
plt.show()

