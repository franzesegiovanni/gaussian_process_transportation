"""
Authors: Ravi Prakash and Giovanni Franzese, Dec 2022
Email: r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import matplotlib.pyplot as plt
from pynput import keyboard
from pynput.keyboard import Listener,KeyCode
import numpy as np
import pathlib
# import rospy
# from sensor_msgs.msg import  Joy
# from geometry_msgs.msg import Point, WrenchStamped, PoseStamped, Vector3

class Drawing():
    def __init__(self, draw=2):
        self.fig, self.ax =  plt.subplots()
        self.ax.set_xlim(-50, 50-1)
        self.ax.set_ylim(-50, 50-1)
        self.points, = self.ax.plot([], [], 'o')

        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
         # Start keyboard listener
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()
        self.keep_drawing = False
        self.idx=0
        self.x = []
        self.y = []
        self.z = []

        self.control_freq=100 # [Hz]
        self.draw=draw
        if self.draw==2:
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_move2D)
            plt.show()


    
    def _on_press(self, key):

        if key == KeyCode.from_char('z'):
            self.keep_drawing = True
        elif key == KeyCode.from_char('w'):
            print("Saving the source surface")
            x= self.x[self.idx:]
            y=self.y[self.idx:]
            z=self.z[self.idx:]
            if self.draw==2:
                self.floor = np.array([x,y]).T
            elif self.draw==3:
                self.floor = np.array([x,y,z]).T
            self.idx=len(self.x)
            self.keep_drawing = False
        elif key == KeyCode.from_char('d'):
            print("Saveing the demo")
            x= self.x[self.idx:]
            y=self.y[self.idx:]
            self.demo = np.array([x,y]).T
            self.idx=len(self.x)
            self.keep_drawing = False
        elif key == KeyCode.from_char('n'):
            print("Saving the target surface")
            x= self.x[self.idx:]
            y=self.y[self.idx:]
            self.newfloor = np.array([x,y]).T
            self.idx=len(self.x)
            self.keep_drawing = False
      

    def _on_move2D(self, event):
        # append event's data to lists
        if self.keep_drawing:
            self.x.append(event.xdata)
            self.y.append(event.ydata)        
            # update plot's data  
            self.points.set_data(self.x,self.y)
            # restore background
            self.fig.canvas.restore_region(self.background)
            # redraw just the points
            self.ax.draw_artist(self.points)
            # fill in the axes rectangle
            self.fig.canvas.blit(self.ax.bbox)


    def save(self, data='last'):
        np.savez(str(pathlib.Path().resolve())+'/data/'+str(data)+'.npz', 
        demo=self.demo, 
        floor=self.floor, 
        newfloor=self.newfloor) 

