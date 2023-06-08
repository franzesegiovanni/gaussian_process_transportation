"""
Authors: Ravi Prakash & Giovanni Franzese, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

from ILoSA import ILoSA
from GILoSA import Transport
from GILoSA.tag_detector import Tag_Detector
from GILoSA.surface_pointcloud_detector import Surface_PointCloud_Detector
from GILoSA.read_pose_arm import LeftArmPose
import rospy


class GILoSA_surface(Transport,Surface_PointCloud_Detector, ILoSA):
    def __init__(self):
        rospy.init_node('GILoSA', anonymous=True)
        rospy.sleep(2)
        super(GILoSA_surface,self).__init__()
        
class GILoSA_arm(Transport, LeftArmPose, ILoSA):
    def __init__(self):
        rospy.init_node('GILoSA', anonymous=True)
        rospy.sleep(2)
        super(GILoSA_arm,self).__init__()        
        
class GILoSA(Transport, Tag_Detector, ILoSA):
    def __init__(self):
        rospy.init_node('GILoSA', anonymous=True)
        rospy.sleep(2)
        super(GILoSA,self).__init__()
        
