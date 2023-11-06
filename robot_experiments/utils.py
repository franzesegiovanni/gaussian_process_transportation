from tf.transformations import euler_from_quaternion, quaternion_multiply, quaternion_inverse
from geometry_msgs.msg import Pose
import numpy as np  
import itertools
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

def sort_points(points):
    permutations = list(itertools.permutations(points))
    cost = []
    for perm in permutations:
        perm= np.array(perm)
        cost.append(np.sum(np.linalg.norm(perm[1:,:]-perm[:-1,:],1)))
    min_index = np.argmin(cost)
    permuted_list = permutations[min_index]
    index = []
    for i in range(len(points)):
        index.append(int(np.argmin(np.linalg.norm(permuted_list-points[i], axis=1))))
    return permuted_list, index

def calculate_elbow_angle(shoulder, elbow, wrist):

    # Calculate the upper arm and forearm vectors
    upper_arm_vector = elbow - shoulder
    forearm_vector = wrist - elbow

    # Normalize the vectors
    normalized_upper_arm = upper_arm_vector / np.linalg.norm(upper_arm_vector)
    normalized_forearm = forearm_vector / np.linalg.norm(forearm_vector)

    # Calculate the dot product of the normalized vectors
    dot_product = np.dot(normalized_upper_arm, normalized_forearm)

    # Calculate the angle in radians
    angle_radians = np.arccos(dot_product)

    # Convert the angle to degrees
    angle_degrees = 180- np.degrees(angle_radians)

    return angle_degrees

def calculate_elbow_angle2(shoulder, elbow, wrist):

    # Calculate the upper arm and forearm vectors
    upper_arm_vector = elbow - shoulder
    forearm_vector = wrist - shoulder

    # Normalize the vectors
    normalized_upper_arm = upper_arm_vector / np.linalg.norm(upper_arm_vector)
    normalized_forearm = forearm_vector / np.linalg.norm(forearm_vector)

    # Calculate the dot product of the normalized vectors
    dot_product = np.dot(normalized_upper_arm, normalized_forearm)

    # Calculate the angle in radians
    angle_radians = np.arccos(dot_product)

    # Convert the angle to degrees
    angle_degrees = 180- np.degrees(angle_radians)

    return angle_degrees