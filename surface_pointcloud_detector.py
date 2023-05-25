"""
Authors: Ravi Prakash and Giovanni Franzese, May 2023
Email: r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from regressor3d import GPR

class Surface_PointCloud_Detector(): 
    def __init__(self,queue_size =5):
        super(Surface_PointCloud_Detector, self).__init__()    
        rospy.init_node('pointcloud', anonymous=True)   
        rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pointcloud_subscriber_callback)
        self.point_cloud = o3d.geometry.PointCloud()
        self.source_distribution = None
        self.target_distribution = None
        self.source_mesh = None
        self.target_mesh = None


    def pointcloud_subscriber_callback(self, msg):

        # Convert PointCloud2 message to numpy array
        pc_data = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))

        # Create Open3D point cloud from numpy array
        self.point_cloud.points = o3d.utility.Vector3dVector(pc_data)


    def pick_points(self,pcd):
        print("")
        print("1) Please pick the corner points of the PCD using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        print("selected points:",vis.get_picked_points())
        return vis.get_picked_points()

    def meshgrid(self,picked_points):
        # Define the 4 corner points of the quadrilateral
        A = np.array(picked_points[0,:-1])
        B = np.array(picked_points[1,:-1])
        D = np.array(picked_points[2,:-1])
        C = np.array(picked_points[3,:-1])

        # Define the number of points in the x and y directions of the grid
        nx = 50
        ny = 50

        # Create the linspace arrays for AB and CD
        AB = np.linspace(A, B, nx)
        CD = np.linspace(C, D, nx)

        # Initialize the meshgrid array
        meshgrid = np.zeros((nx, ny, 2))

        # Create the meshgrid by linearly interpolating between AB and CD
        for i in range(nx):
            meshgrid[i, :, :] = np.linspace(AB[i], CD[i], ny)

        return meshgrid


    
    def crop_geometry(self,pcd):

        print("Save a PCD corresponding to the item")
        print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
        print("2) Press 'K' to lock screen and to switch to selection mode")
        print("3) Drag for rectangle selection,")
        print("   or use ctrl + left click for polygon selection")
        print("4) Press 'C' to get a selected geometry and to save it")
        print("5) Press 'F' to switch to freeview mode")
        vis = o3d.visualization.draw_geometries_with_editing([pcd])


    def record_distribution(self, distribution):
        
        print("Visualize np Distribution and Grid")
        picked_id_distribution = self.pick_points(distribution)
        picked_points_distribution = np.asarray(distribution.points)[picked_id_distribution]
        meshgrid_distribution = self.meshgrid(picked_points_distribution)

        down_distribution = distribution.voxel_down_sample(voxel_size=0.1)

        distribution_np = np.asarray(down_distribution.points)
        fig = plt.figure()
        ax = plt.axes(projection ='3d') 
        ax.scatter(distribution_np[:,0], distribution_np[:,1], distribution_np[:,2])
        plt.show()

        k_distribution = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(2)) + WhiteKernel(0.01 )
        gp_distribution = GPR(kernel=k_distribution)
        gp_distribution.fit(distribution_np[:,:2],distribution_np[:,2].reshape(-1,1))

        # newgrid = newgrid.reshape(-1,2)
        newZ,_ = gp_distribution.predict(meshgrid_distribution.reshape(-1,2))
        distribution_surface = np.array([meshgrid_distribution[:,:,0],meshgrid_distribution[:,:,1],newZ.reshape(meshgrid_distribution[:,:,0].shape)]).T
        distribution_surface = distribution_surface.reshape(-1,3)
        distribution_surface=distribution_surface[::3]
        return distribution_surface





    def record_source_distribution(self):
        rospy.sleep(1)
        source_cloud = self.point_cloud
        self.crop_geometry(source_cloud)
        source_cropped_cloud = o3d.io.read_point_cloud("./source.ply")
        self.source_distribution = self.record_distribution(source_cropped_cloud)


    def record_target_distribution(self):
        rospy.sleep(1)
        target_cloud = self.point_cloud
        self.crop_geometry(target_cloud)
        target_cropped_cloud = o3d.io.read_point_cloud("./target.ply")
        self.target_distribution = self.record_distribution(target_cropped_cloud)
