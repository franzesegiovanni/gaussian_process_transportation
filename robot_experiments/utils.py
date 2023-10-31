from tf.transformations import euler_from_quaternion, quaternion_multiply, quaternion_inverse
from geometry_msgs.msg import Pose
import numpy as np  
def relative_transformation(pose, relative2pose):
    pose_relative=Pose()
    pose_relative.position.x= pose.position.x - relative2pose.position.x
    pose_relative.position.y= pose.position.y - relative2pose.position.y
    pose_relative.position.z= pose.position.z - relative2pose.position.z
    # Convert rotation to NumPy array
    rotation1 = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    rotation2 = np.array([relative2pose.orientation.x, relative2pose.orientation.y, relative2pose.orientation.z, relative2pose.orientation.w])
    # Calculate relative rotation as a quaternion
    relative_quat = quaternion_multiply(rotation1, quaternion_inverse(rotation2))
    #compute row, pitch, yaw
    (roll, pitch, yaw) = euler_from_quaternion(relative_quat)
    pose_relative.orientation.x=roll
    pose_relative.orientation.y=pitch
    pose_relative.orientation.z=yaw
    return pose_relative