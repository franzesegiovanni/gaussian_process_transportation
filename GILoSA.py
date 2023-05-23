"""
Authors: Ravi Prakash & Giovanni Franzese, March 2023
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

from ILoSA import ILoSA
from transport import Transport 
from tag_detector import Tag_Detector
import rospy

class GILoSA(Transport, Tag_Detector, ILoSA):
    def __init__(self):
        rospy.init_node('GILOSA', anonymous=True)
        rospy.sleep(2)
        super(GILoSA,self).__init__()
        
