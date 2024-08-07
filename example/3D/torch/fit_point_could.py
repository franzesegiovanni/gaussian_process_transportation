import pathlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from policy_transportation.models.torch.stocastic_variational_gaussian_process import StocasticVariationalGaussianProcess
import os
files=['dustbin_cover_point_cloud_distribution', 'pan_point_cloud_distribution', 'white_towelholder_point_cloud_distribution', 'wood_plate_point_cloud_distribution']
file_dir=os.path.dirname(__file__)

for file in files:
    print("Load the point cloud distribution of the object: ", file)
    distribution_np =np.load(file_dir+'/data/'+file+'.npz')['point_cloud_distribution']

    fig = plt.figure()
    ax = plt.axes(projection ='3d') 
    ax.scatter(distribution_np[:,0], distribution_np[:,1], distribution_np[:,2])

    print("Find the points corresponding of the selected grid")
    gp_distribution=StocasticVariationalGaussianProcess(distribution_np[:,:2], distribution_np[:,2].reshape(-1,1), num_inducing=1000)
    gp_distribution.fit(num_epochs=20) 

    x_min=distribution_np[:,0].min()
    x_max=distribution_np[:,0].max()
    y_min=distribution_np[:,1].min()
    y_max=distribution_np[:,1].max()
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    meshgrid_distribution = np.array(np.meshgrid(x, y)).T.reshape(-1,2)

    newZ = gp_distribution.predict(meshgrid_distribution)

    distribution_surface=np.hstack([meshgrid_distribution,newZ.reshape(-1,1)])
    # print("distribution_surface.shape",distribution_surface.shape)
    ax.scatter(distribution_surface[:,0], distribution_surface[:,1], distribution_surface[:,2], 'r')
plt.show()