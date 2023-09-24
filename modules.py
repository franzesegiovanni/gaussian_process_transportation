"""
Authors: Ravi Prakash & Giovanni Franzese, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

from ILoSA import ILoSA
from policy_transportation import GaussianProcessTransportation as Transport
from sensors.tag_detector import Tag_Detector
from sensors.surface_pointcloud_detector import Surface_PointCloud_Detector
from sensors.read_pose_arm import LeftArmPose
import rospy


class Surface_Trasportation(Transport,Surface_PointCloud_Detector, ILoSA):
    def __init__(self):
        rospy.init_node('GILoSA', anonymous=True)
        rospy.sleep(2)
        super(Surface_Trasportation,self).__init__()
        
class Dressing_Trasportation(Transport, LeftArmPose, ILoSA):
    def __init__(self):
        rospy.init_node('GILoSA', anonymous=True)
        rospy.sleep(2)
        super(Dressing_Trasportation,self).__init__()        
        
class Tag_Transportation(Transport, Tag_Detector, ILoSA):
    def __init__(self):
        rospy.init_node('GILoSA', anonymous=True)
        rospy.sleep(2)
        super(Tag_Transportation,self).__init__()


        
