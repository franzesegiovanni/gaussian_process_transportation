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
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from geometry_msgs.msg import PoseStamped
import open3d as o3d
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from GILoSA import GaussianProcess

class Surface_PointCloud_Detector(): 
    def __init__(self):
        super(Surface_PointCloud_Detector, self).__init__() 

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.base_frame = "panda_link0"

        # camera and tags
        self.view_marker = PoseStamped()

        # For Cleaning experiment
        self.view_marker.header.frame_id = "panda_link0"
        self.view_marker.pose.position.x = 0.4585609490798406
        self.view_marker.pose.position.y = -0.04373075604079746
        self.view_marker.pose.position.z = 0.6862181406538658
        self.view_marker.pose.orientation.w = 0.03724672277393113
        self.view_marker.pose.orientation.x =  0.9986700943869168
        self.view_marker.pose.orientation.y =  0.03529063207161849
        self.view_marker.pose.orientation.z = -0.004525063460755314


        rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pointcloud_subscriber_callback)
        self.point_cloud = o3d.geometry.PointCloud()
        self.source_distribution = None
        self.target_distribution = None



    def pointcloud_subscriber_callback(self, msg):
        try:
            # Retrieve the transform between the camera frame and the robot frame

            transform = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(1.0))

            # Convert the point cloud to the robot frame
            msg_in_robot_frame = do_transform_cloud(msg, transform)

            # Convert PointCloud2 message to numpy array
            pc_data = pc2.read_points(msg_in_robot_frame, skip_nans=True, field_names=("x", "y", "z"))
            # pc_data = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))

            # Create Open3D point cloud from numpy array
            self.point_cloud.points = o3d.utility.Vector3dVector(pc_data)
            # self.point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # Flip the pointclouds, otherwise they will be upside down. 

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr('Error occurred during point cloud transformation: %s', str(e))


        
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



    def crop_geometry(self, picked_points_distribution, pcd):
        min_bound = np.min(picked_points_distribution, axis=0)
        max_bound = np.max(picked_points_distribution, axis=0)
    
        # Set z-bounds to positive and negative infinity
        min_bound[2] = -np.inf
        max_bound[2] = np.inf
    
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

        # Crop the point cloud using the bounding box
        cropped_pcd = pcd.crop(bbox)

        return cropped_pcd

    def record_distribution(self, distribution):
        
        print("Visualize np Distribution and Grid")
        picked_id_distribution = self.pick_points(distribution)
        picked_points_distribution = np.asarray(distribution.points)[picked_id_distribution]

        cropped_distribution = self.crop_geometry(picked_points_distribution, distribution)

        meshgrid_distribution = self.meshgrid(picked_points_distribution)

        down_distribution = cropped_distribution.voxel_down_sample(voxel_size=0.02)

        distribution_np = np.asarray(down_distribution.points)
        fig = plt.figure()
        ax = plt.axes(projection ='3d') 
        ax.scatter(distribution_np[:,0], distribution_np[:,1], distribution_np[:,2])
        plt.show()

        k_distribution = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(2)) + WhiteKernel(0.01 )
        gp_distribution = GaussianProcess(kernel=k_distribution)
        gp_distribution.fit(distribution_np[:,:2],distribution_np[:,2].reshape(-1,1))

        # newgrid = newgrid.reshape(-1,2)
        newZ,_ = gp_distribution.predict(meshgrid_distribution.reshape(-1,2))
        distribution_surface = np.array([meshgrid_distribution[:,:,0],meshgrid_distribution[:,:,1],newZ.reshape(meshgrid_distribution[:,:,0].shape)]).T
        distribution_surface = distribution_surface.reshape(-1,3)
        distribution_surface=distribution_surface[::3]
        return distribution_surface

    def convert_distribution_to_array(self):
        pass



    def record_source_distribution(self):
        rospy.sleep(5)
        source_cloud = self.point_cloud
        self.source_distribution = self.record_distribution(source_cloud)


    def record_target_distribution(self):
        rospy.sleep(1)
        target_cloud = self.point_cloud
        self.target_distribution = self.record_distribution(target_cloud)
