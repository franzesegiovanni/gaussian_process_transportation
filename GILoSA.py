"""
Authors: Ravi Prakash & Giovanni Franzese, March 2023
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

import ILoSA
from transport import Transport 
from tag_detector import Tag_Detector
import rospy

class GILoSA(ILoSA, Transport, Tag_Detector):
    def __init__(self):
        rospy.init_node('GILOSA', anonymous=True)
        rospy.sleep(2)
        super().__init__()
        
