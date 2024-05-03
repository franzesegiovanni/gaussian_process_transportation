"""
Authors: Ravi Prakash & Giovanni Franzese, March 2023
Email: g.franzese@tudelft.nl, r.prakash@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation import GaussianProcessTransportation as Transport
from sensors.tag_detector import Tag_Detector
from sensors.surface_pointcloud_detector import Surface_PointCloud_Detector
from SIMPLe import SIMPLe
import rospy

class GPT_surface(Transport,Surface_PointCloud_Detector,SIMPLe):
    def __init__(self):
        rospy.init_node('GPT', anonymous=True)
        rospy.sleep(2)
        super(GPT_surface,self).__init__()
              
        
class GPT_tag(Transport, Tag_Detector, SIMPLe):
    def __init__(self):
        rospy.init_node('GPT', anonymous=True)
        rospy.sleep(2)
        super(GPT_tag,self).__init__()


        
