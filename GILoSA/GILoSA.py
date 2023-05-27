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
import rospy


# class GILoSA(ILoSA, Transport, Surface_PointCloud_Detector):
#     def __init__(self):
#         rospy.init_node('GILoSA', anonymous=True)
#         rospy.sleep(2)
#         super(GILoSA,self).__init__()
        
# class GILoSA(ILoSA, Transport, LeftArmPose):
#     def __init__(self):
#         rospy.init_node('GILoSA', anonymous=True)
#         rospy.sleep(2)
#         super(GILoSA,self).__init__()        
        
class GILoSA(ILoSA, Transport, Tag_Detector):
    def __init__(self):
        rospy.init_node('GILoSA', anonymous=True)
        rospy.sleep(2)
        super(GILoSA,self).__init__()
        
