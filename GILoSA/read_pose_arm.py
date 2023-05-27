"""
Authors: Giovanni Franzese & Ravi Prakash, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import cv2
import mediapipe as mp
import rospy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
import numpy as np
import cv2
import struct

class LeftArmPose():
    def __init__(self):
        rospy.init_node("arm_pose_calculation")    
        # Initialize the Pose module
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0, min_tracking_confidence=0)

        # Create a figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.point_cloud_sub = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.pointcloud_callback)

        self.image_pub= rospy.Publisher('/image_pose', Image, queue_size=1)

        self.point_step = 32
        self.row_step = 20480
    def image_callback(self,msg):
        # Convert ROS Image message to OpenCV image
        self.curr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # print(cv2.__version__)
        # print("self.curr_img",self.curr_img.shape)

    def depth_callback(self,msg):
        # Convert ROS Image message to OpenCV image
        self.curr_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")  

    def pointcloud_callback(self, msg):
        self.point_cloud=msg.data
        self.point_step = msg.point_step
        self.row_step = msg.row_step
        self.height = msg.height
        self.width = msg.width
        self.step=msg.point_step
       
      
    def pixel_to_3D(self, px, py):
        offset = py*self.row_step + px*self.point_step
        x, y, z = float('nan'), float('nan'), float('nan')
        try:
            x = np.frombuffer(bytes([self.point_cloud[int(0+offset)],self.point_cloud[int(1+offset)],self.point_cloud[int(2+offset)],self.point_cloud[int(3+offset)]]),dtype=np.float32)
            y = np.frombuffer(bytes([self.point_cloud[int(4+offset)],self.point_cloud[int(5+offset)],self.point_cloud[int(6+offset)],self.point_cloud[int(7+offset)]]),dtype=np.float32)
            z = np.frombuffer(bytes([self.point_cloud[int(8+offset)],self.point_cloud[int(9+offset)],self.point_cloud[int(10+offset)],self.point_cloud[int(11+offset)]]),dtype=np.float32)
        except:
             print('Point cloud does like it')
        return x,y,z
    
    def extract_arm_landmarks(self, results):
        # Extract the landmarks for the left and right arms)
        if results.pose_landmarks!=None:
            # print(len(results.pose_landmarks.landmark))
            shoulder=results.pose_landmarks.landmark[11]
            shoulder_pixel=[int(shoulder.x*self.w), int(shoulder.y*self.h)]
            # print(self.curr_depth.shape)
            # self.curr_depth[int(self.shoulder_pixel[0]*self.w), int(self.shoulder_pixel[1]*self.h), :]

            # print("Shoulder:", shoulder)
            elbow=results.pose_landmarks.landmark[13]
            elbow_pixel=[int(elbow.x*self.w), int(elbow.y*self.h)]
            # print("Elbow:", elbow)
            hand=results.pose_landmarks.landmark[15]
            hand_pixel=[int(hand.x*self.w), int(hand.y*self.h)]
            # print("Hand:", hand)    
        return shoulder_pixel, elbow_pixel, hand_pixel

    def read_pose(self):
        color = (0, 0, 255)  # BGR format (blue, green, red)
        color_1 = (0, 255, 0)  # BGR format (blue, green, red)
        # Choose dot radius
        radius = 10
        thickness=5
        while True:
            # Process the grayscale image with MediaPipe to extract the pose landmarks
            image=self.curr_img
            results = self.pose.process(image)
            
            self.h, self.w = image.shape[:2]

           
            # # Extract the arm landmarks from the pose landmarks
            if results.pose_landmarks!=None:
                shoulder_pixel, elbow_pixel, hand_pixel= self.extract_arm_landmarks(results)
                x_shou,y_shou,z_shou = self.pixel_to_3D(shoulder_pixel[0],shoulder_pixel[1])
                x_elbow,y_elbow,z_elbow = self.pixel_to_3D(elbow_pixel[0],elbow_pixel[1])
                x_hand,y_hand,z_hand = self.pixel_to_3D(hand_pixel[0],hand_pixel[1])

                print("Shoulder:", x_shou, y_shou, z_shou)
                print("elbow:", x_elbow,y_elbow,z_elbow )
                print("hand:", x_hand,y_hand,z_hand)    
                cv2.circle(image, shoulder_pixel, radius, color, -1)
                cv2.line(image, shoulder_pixel, elbow_pixel, color_1, thickness)
                cv2.circle(image, elbow_pixel, radius, color, -1)
                cv2.line(image, elbow_pixel, hand_pixel, color_1, thickness)
                cv2.circle(image, hand_pixel, radius, color, -1)

                try:
                    ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
                except CvBridgeError as e:
                    print(e)

                self.image_pub.publish(ros_image)
    


            

if __name__ == '__main__':

    arm=LeftArmPose()
    rospy.sleep(2)
    arm.read_pose()
