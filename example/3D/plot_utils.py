import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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
                u[i,j,k]=vel[0][0]
                v[i,j,k]=vel[0][1]
                w[i,j,k]=vel[0][2]
    ax = plt.figure().add_subplot(projection='3d')        
    ax.quiver(dataXX, dataYY, dataZZ, u, v,w, normalize=True)
    ax.scatter(demo[:,0],demo[:,1],demo[:,2], color=[1,0,0])
    # ax.scatter(surface[:,0],surface[:,1], color=[0,0,0])
    # plt.show() 

   

def plot_vector_field_minvar3D(model,datax_grid,datay_grid,dataz_grid,demo,surface):
    dataXX, dataYY, dataZZ = np.meshgrid(datax_grid, datay_grid, dataz_grid)
    u=np.ones((len(datax_grid),len(datay_grid),len(dataz_grid)))
    v=np.ones((len(datax_grid),len(datay_grid),len(dataz_grid)))
    w=np.ones((len(datax_grid),len(datay_grid),len(dataz_grid)))
    for i in range(len(datax_grid)):
        for j in range(len(datay_grid)):
            for k in range(len(dataz_grid)):
                pos=np.array([dataXX[i,j,k],dataYY[i, j,k],dataZZ[i,j,k]]).reshape(1,-1)
                [vel, std]=model.predict(pos)
                [_,grad]=model.derivative(pos)
                u[i,j,k]=vel[0][0]-std[0]*grad[0,0,0]/np.sqrt(grad[0,0,0]**2+grad[0,1,0]**2+grad[0,2,0]**2)
                v[i,j,k]=vel[0][1]-std[0]*grad[0,1,0]/np.sqrt(grad[0,0,0]**2+grad[0,1,0]**2+grad[0,2,0]**2)
                w[i,j,k]=vel[0][2]-std[0]*grad[0,2,0]/np.sqrt(grad[0,0,0]**2+grad[0,1,0]**2+grad[0,2,0]**2)
        

    ax = plt.figure().add_subplot(projection='3d')        
    ax.quiver(dataXX, dataYY, dataZZ, u, v,w)
    ax.scatter(demo[:,0],demo[:,1],demo[:,2], color=[1,0,0])
    # ax.scatter(surface[:,0],surface[:,1], color=[0,0,0])
    # plt.show() 

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