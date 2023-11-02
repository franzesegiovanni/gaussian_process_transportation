# Load the LASA data
from load_data import _PyLasaDataSet
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from policy_transportation import GaussianProcess as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from policy_transportation import GaussianProcessTransportation as Transport
DataSet = _PyLasaDataSet()
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
def surface_equations(u, v):
    u_min= np.min(u)
    u_max= np.max(u)
    v_min= np.min(v)
    v_max= np.max(v)
    u_mean= (u_min+u_max)/2
    v_mean= (v_min+v_max)/2

    x = u
    y = v
    z =1+ ((u-u_mean)/(u_max-u_mean))**2 - ((v-v_mean)/(u_max-u_mean))**2  # Modify the equation to create your desired surface
    return x, y, z

current_file_directory = os.path.dirname(os.path.realpath(__file__))
angle_data = DataSet.BendedLine
demos = angle_data.demos # list of 7 Demo objects, each corresponding to a 
                         # repetition of the pattern
downsample=10
X=np.transpose(demos[0].pos[:,100::downsample])
X=np.hstack((X, np.zeros_like(X[:,[0]]))) 
X_dot=np.zeros((len(X),3))
for j in range(len(X)-1): 
    X_dot[j,:]=(X[j+1,:]-X[j,:])
    X_dot[:-1,-1]=-0.03*np.ones_like(X_dot[:-1,-1])

for i in range(1,7):
    X_traj=np.transpose(demos[i].pos[:,100::downsample])
    X_traj=np.hstack((X_traj, np.zeros_like(X_traj[:,[0]]))) #add z coordinate
    X=np.vstack((X, X_traj))
    X_dot_traj=np.zeros((len(X_traj),3))
    for j in range(len(X_traj)-1): 
        X_dot_traj[j,:]=np.transpose(X_traj[j+1,:]-X_traj[j,:])
        X_dot_traj[:-1,-1]=-0.03*np.ones_like(X_dot_traj[:-1,-1])
    X_dot=np.vstack((X_dot, X_dot_traj))
      

k = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(3), nu=2.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp=GPR(kernel=k)
gp.fit(X, X_dot)    

x_grid=np.linspace(np.min(X[:,0]-2), np.max(X[:,0]+2), 20)
y_grid=np.linspace(np.min(X[:,1]-2), np.max(X[:,1]+2), 20)
# Creating grids
XX, YY = np.meshgrid(x_grid, y_grid)
XX_flat=XX.flatten()
YY_flat=YY.flatten()
ZZ_flat=np.zeros_like(XX_flat)
X_pred=np.vstack((XX_flat, YY_flat, ZZ_flat)).T

mean, std=gp.predict(X_pred)
[_,grad]=gp.derivative(X_pred)
vect=np.sqrt(grad[:,0,0]**2+grad[:,1,0]**2+grad[:,2,0]**2)
dX=mean[:,0]-2*std[:,0]*grad[:,0,0]/vect
dY=mean[:,1]-2*std[:,1]*grad[:,1,0]/vect
dZ=mean[:,2]#-2*std*grad[:,2,0]/vect
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = plt.figure().add_subplot(projection='3d')
ax.quiver(XX_flat, YY_flat, ZZ_flat, dX, dY, dZ, color='b', arrow_length_ratio=0.08)
ax.scatter(X[:,0], X[:,1], X[:,2], color='r')
# ax.set_zlim(-0.2, 0.2)

source_z=np.zeros_like(XX)
source_x, source_y= XX, YY
ax.plot_surface(source_x, source_y, source_z, color='g', alpha=0.5)
ax.plot3D(source_x[:, 0], source_y[:, 0], source_z[:, 0], color='g', alpha=0.6, linewidth=2)
ax.plot3D(source_x[:, -1], source_y[:, -1], source_z[:, -1], color='g', alpha=0.6, linewidth=2)
ax.plot3D(source_x[0, :], source_y[0, :], source_z[0, :], color='g', alpha=0.6, linewidth=2)
ax.plot3D(source_x[-1, :], source_y[-1, :], source_z[-1, :], color='g', alpha=0.6, linewidth=2)
source_distribution=np.vstack((source_x.flatten(), source_y.flatten(), source_z.flatten())).T
ax.axis('off')


# Plot the target surface

# Create the data for the surface
target_x, target_y, target_z = surface_equations(XX , YY)

# Plot the surface
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(target_x, target_y, target_z, color='b', alpha=0.5, linewidths=0)
ax.plot3D(target_x[:, 0], target_y[:, 0], target_z[:, 0], color='b', alpha=0.6, linewidth=2)
ax.plot3D(target_x[:, -1], target_y[:, -1], target_z[:, -1], color='b', alpha=0.6, linewidth=2)
ax.plot3D(target_x[0, :], target_y[0, :], target_z[0, :], color='b', alpha=0.6, linewidth=2)
ax.plot3D(target_x[-1, :], target_y[-1, :], target_z[-1, :], color='b', alpha=0.6, linewidth=2)
target_distribution=np.stack((target_x.flatten(), target_y.flatten(), target_z.flatten())).T
# ax.axis('off')

kernel=C*RBF(1*np.ones(3)) + WhiteKernel(0.01 ) #this kernel works much better!
transport=Transport()
transport.source_distribution=source_distribution 
transport.target_distribution=target_distribution
transport.training_traj=X
transport.training_delta=X_dot
transport.fit_transportation()
transport.apply_transportation()

X1=transport.training_traj
X1_dot=transport.training_delta 

gp_deltaX1=GPR(kernel=k)
gp_deltaX1.fit(X1, X1_dot)

X1_pred=transport.target_distribution
mean, _=gp_deltaX1.predict(X1_pred)
[_,grad]=gp.derivative(X1_pred)
vect=np.sqrt(grad[:,0,0]**2+grad[:,1,0]**2+grad[:,2,0]**2)
dX=mean[:,0]-2*std[:,0]*grad[:,0,0]/vect
dY=mean[:,1]-2*std[:,1]*grad[:,1,0]/vect
dZ=mean[:,2]#-2*std*grad[:,2,0]/vect
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax = plt.figure().add_subplot(projection='3d')
ax.quiver(target_distribution[:,0], target_distribution[:,1], target_distribution[:,2], dX, dY, dZ, color='k', arrow_length_ratio=0.08)
ax.scatter(X1[:,0], X1[:,1], X1[:,2], color='r')
ax.view_init(elev=36, azim=-80)
# ax.set_zlim(-0.2, 0.2)
plt.savefig(current_file_directory+'/pdf/LASE_deformed.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()