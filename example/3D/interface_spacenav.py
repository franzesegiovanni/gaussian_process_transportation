"""
Authors: Ravi Prakash and Giovanni Franzese, Dec 2022
Email: r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import  Joy
from geometry_msgs.msg import  Vector3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pathlib
class Spacenav():

    def __init__(self):

        self.control_freq=100 # [Hz]

        self.start = False
        self.end = False
        self.feedback = [0, 0, 0]
        self.x = []
        self.y= []
        self.z = []
        self.ax = plt.axes(projection='3d')
        # self.ax.view_init(elev=10., azim=180)


    # spacemouse joystick subscriber
    def teleop_callback(self, data):
        self.feedback = [data.x, data.y, data.z]


    # spacemouse buttons subscriber
    def btns_callback(self, data):
        self.left_btn = data.buttons[0]
        self.right_btn = data.buttons[1]
        if self.left_btn ==1:
            self.start = True
        elif self.right_btn == 1:
            self.end = True
        

    def connect_ROS(self):
        rospy.init_node('interface_spacenav', anonymous=True)
        self.r=rospy.Rate(self.control_freq)

        rospy.Subscriber("/spacenav/offset", Vector3, self.teleop_callback)
        rospy.Subscriber("/spacenav/joy", Joy, self.btns_callback)

 

    def save(self, data='3dlast'):
        np.savez(str(pathlib.Path().resolve())+'/data/'+str(data)+'.npz', 
        demo=self.demo) 
        print("saved")

    def plot_traj(self):
        while True:

            if self.start == True:    
                self.x.append(self.feedback[0]*100)
                self.y.append(self.feedback[1]*100)  
                self.z.append(self.feedback[2]*100)  
                plt.cla()
                self.ax.set_xlim(-50, 50)
                self.ax.set_ylim(-50, 50)
                self.ax.set_zlim(-50, 50)
                self.ax.scatter(self.x, self.y, self.z, c = 'r', marker='o')
                self.ax.set_xlabel('X-axis')
                self.ax.set_ylabel('Y-axis')
                self.ax.set_zlabel('Z-axis')
                plt.draw()
                plt.pause(0.001)
            if self.end == True:
                self.demo = np.array([self.x,self.y,self.z]).T
                
                self.save()
                break
        print("Exiting plot traj")
