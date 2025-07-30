import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from policy_transportation import GaussianProcess as GPR
def plot_vector_field(X, deltaX, distribution, min_var=False):
    kernel_policy = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01) 
    model=GPR(kernel=kernel_policy)
    model.fit(X, deltaX)
    x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
    y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
    dataXX, dataYY = np.meshgrid(x_grid, y_grid)
    pos_array= np.column_stack((dataXX.ravel(), dataYY.ravel()))
    [vel, std]=model.predict(pos_array, return_std=True)
    if min_var: 
        grad=model.derivative_of_variance(pos_array).transpose()
        vel=vel-2*std*grad/np.linalg.norm(grad, axis=1).reshape(-1,1)
    u= vel[:,0].reshape(dataXX.shape)
    v= vel[:,1].reshape(dataXX.shape)
    fig = plt.figure(figsize = (12, 7))
    plt.streamplot(dataXX, dataYY, u, v, density = 2)
    plt.scatter(X[:,0],X[:,1], color=[1,0,0])
    plt.scatter(distribution[:,0],distribution[:,1], color=[0,0,0])


def plot_traj_evolution(model,x_grid,y_grid,z_grid,demo, surface):
    start_pos = np.random.uniform([x_grid[0], y_grid[0], z_grid[0]], [x_grid[-1], y_grid[-1], z_grid[-1]], size=(1, 3))
    traj = np.zeros((1000,3))
    pos=np.array(start_pos).reshape(1,-1)   
    for i in tqdm(range(1000)):
        pos=np.array(pos).reshape(1,-1)

        [vel, std]=model.predict(pos, return_std=True)
        grad=model.derivative_of_variance(pos)
        f_stable=np.array([grad[0,0],grad[1,0],grad[2,0]])/np.sqrt(grad[0,0]**2+grad[1,0]**2+grad[2,0]**2)
        pos = pos+vel.reshape(1,-1)-std[0]*f_stable

        traj[i,:]= pos


    ax = plt.figure().add_subplot(projection='3d')    
    ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2], cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)    
    ax.scatter(demo[:,0],demo[:,1],demo[:,2], color=[1,0,0])
    ax.scatter(traj[:,0],traj[:,1],traj[:,2], color=[0,0,1])

def plot_traj_3D(trajectory, surface):

    ax = plt.figure().add_subplot(projection='3d')    
    ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2], cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)    
    ax.scatter(trajectory[:,0],trajectory[:,1],trajectory[:,2], color=[0,0,1])

def draw_error_band(ax, x, y, err, loop=False, **kwargs):
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # end points of errors
    xp = x + nx * err[:,0]
    yp = y + ny * err[:,1]
    xn = x - nx * err[:,0]
    yn = y - ny * err[:,1]

    # print(xp.shape, xn.shape, yp.shape, yn.shape)

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    codes = np.full(len(vertices), Path.LINETO)
    if loop==True:
        codes[0] = codes[len(xp)] = Path.MOVETO
    codes[0] = Path.MOVETO
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, label='Uncertainty',  **kwargs))


# Function to create orientation frames from quaternions
def plot_orientation_frame(ax, position, orientation, length=0.2, skip=10):
    for i in range(0, len(position), skip):
        # Convert quaternion to rotation matrix (w,x,y,z format)
        quat = orientation[i]
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses x,y,z,w format
        
        # Create basis vectors
        basis = np.eye(3) * length
        rotated_basis = rot.apply(basis)
        
        # Get position
        pos = position[i]
        
        # Plot each axis (RGB for XYZ)
        ax.quiver(pos[0], pos[1], pos[2], 
                 rotated_basis[0, 0], rotated_basis[0, 1], rotated_basis[0, 2], 
                 color='red', length=length)
        ax.quiver(pos[0], pos[1], pos[2], 
                 rotated_basis[1, 0], rotated_basis[1, 1], rotated_basis[1, 2], 
                 color='green', length=length)
        ax.quiver(pos[0], pos[1], pos[2], 
                 rotated_basis[2, 0], rotated_basis[2, 1], rotated_basis[2, 2], 
                 color='blue', length=length)
