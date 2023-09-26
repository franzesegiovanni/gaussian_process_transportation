import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import PathPatch
from matplotlib.path import Path

def plot_vector_field(model,datax_grid,datay_grid,demo,surface):
    dataXX, dataYY = np.meshgrid(datax_grid, datay_grid)
    u=np.ones((len(datax_grid),len(datay_grid)))
    v=np.ones((len(datax_grid),len(datay_grid)))
    for i in range(len(datax_grid)):
        for j in range(len(datay_grid)):
            pos=np.array([dataXX[i,j],dataYY[i, j]]).reshape(1,-1)
            [vel, _]=model.predict(pos)
            u[i,j]=vel[0][0]
            v[i,j]=vel[0][1]
    fig = plt.figure(figsize = (12, 7))
    plt.streamplot(dataXX, dataYY, u, v, density = 2)
    plt.scatter(demo[:,0],demo[:,1], color=[1,0,0])
    plt.scatter(surface[:,0],surface[:,1], color=[0,0,0])

def plot_vector_field3D(model,datax_grid,datay_grid,dataz_grid,demo,surface):
    dataXX, dataYY, dataZZ = np.meshgrid(datax_grid, datay_grid, dataz_grid)
    u=np.ones((len(datax_grid),len(datay_grid),len(dataz_grid)))
    v=np.ones((len(datax_grid),len(datay_grid),len(dataz_grid)))
    w=np.ones((len(datax_grid),len(datay_grid),len(dataz_grid)))
    for i in range(len(datax_grid)):
        for j in range(len(datay_grid)):
            for k in range(len(dataz_grid)):
                pos=np.array([dataXX[i,j,k],dataYY[i, j,k],dataZZ[i,j,k]]).reshape(1,-1)
                [vel, var]=model.predict(pos)
                u[i,j,k]=vel[0][0][0]
                v[i,j,k]=vel[0][0][1]
                w[i,j,k]=vel[0][0][2]
    ax = plt.figure().add_subplot(projection='3d')        
    ax.quiver(dataXX, dataYY, dataZZ, u, v,w, density = 2)
    ax.scatter(demo[:,0],demo[:,1],demo[:,2], color=[1,0,0])

def plot_vector_field_minvar(model,datax_grid,datay_grid,demo,surface):
    dataXX, dataYY = np.meshgrid(datax_grid, datay_grid)
    u=np.ones((len(datax_grid),len(datay_grid)))
    v=np.ones((len(datax_grid),len(datay_grid)))
    for i in range(len(datax_grid)):
        for j in range(len(datay_grid)):
            pos=np.array([dataXX[i,j],dataYY[i, j]]).reshape(1,-1)
            [vel, std]=model.predict(pos)
            [_,grad]=model.derivative(pos)
            u[i,j]=vel[0,0]-2*std[0][0]*grad[0,0,0]/np.sqrt(grad[0,0,0]**2+grad[0,0,0]**2)
            v[i,j]=vel[0,1]-2*std[0][0]*grad[0,1,0]/np.sqrt(grad[0,0,0]**2+grad[0,1,0]**2)

    fig = plt.figure(figsize = (12, 7))
    plt.streamplot(dataXX, dataYY, u, v, density = 2)
    plt.scatter(demo[:,0],demo[:,1], color=[1,0,0]) 
    plt.scatter(surface[:,0],surface[:,1], color=[0,0,0])
    plt.title("Minimum variance")


def plot_vector_field_minvar3D(model,datax_grid,datay_grid,demo,surface):
    dataXX, dataYY = np.meshgrid(datax_grid, datay_grid)
    u=np.ones((len(datax_grid),len(datay_grid)))
    v=np.ones((len(datax_grid),len(datay_grid)))
    for i in range(len(datax_grid)):
        for j in range(len(datay_grid)):
            pos=np.array([dataXX[i,j],dataYY[i, j]]).reshape(1,-1)
            [vel, std]=model.predict(pos)
            [_,grad]=model.derivative(pos)
            u[i,j]=vel[0,0]-2*std[0][0]*grad[0,0,0]/np.sqrt(grad[0,0,0]**2+grad[0,0,0]**2)
            v[i,j]=vel[0,1]-2*std[0][0]*grad[0,1,0]/np.sqrt(grad[0,0,0]**2+grad[0,1,0]**2)

    fig = plt.figure(figsize = (12, 7))
    plt.streamplot(dataXX, dataYY, u, v, density = 2)
    plt.scatter(demo[:,0],demo[:,1], color=[1,0,0]) 
    plt.scatter(surface[:,0],surface[:,1], color=[0,0,0])
    plt.title("Minimum variance")

def plot_traj_evolution(model,x_grid,y_grid,z_grid,demo, surface):
    start_pos = np.random.uniform([x_grid[0], y_grid[0], z_grid[0]], [x_grid[-1], y_grid[-1], z_grid[-1]], size=(1, 3))
    traj = np.zeros((1000,3))
    pos=np.array(start_pos).reshape(1,-1)   
    for i in range(1000):
        pos=np.array(pos).reshape(1,-1)

        [vel, std]=model.predict(pos)
        [_,grad]=model.derivative(pos)
        f_stable=np.array([grad[0,0,0],grad[0,1,0],grad[0,2,0]])/np.sqrt(grad[0,0,0]**2+grad[0,1,0]**2+grad[0,2,0]**2)
        pos = pos+vel.reshape(1,-1)-std[0]*f_stable
        # pos = pos+vel.reshape(1,-1)

        traj[i,:]= pos


    ax = plt.figure().add_subplot(projection='3d')    
    newsurf = ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2], cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)    
    ax.scatter(demo[:,0],demo[:,1],demo[:,2], color=[1,0,0])
    ax.scatter(traj[:,0],traj[:,1],traj[:,2], color=[0,0,1])
    # plt.show()

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
    ax.add_patch(PathPatch(path, label='Uncertanites',  **kwargs))