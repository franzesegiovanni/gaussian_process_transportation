import numpy as np
import quaternion
from copy import copy
import pathlib
import tf
import tf2_ros
import rospy
from geometry_msgs.msg import PoseStamped
from apriltag_ros.msg import AprilTagDetectionArray # remeber to souce the workspace with april tags
from tf2_geometry_msgs import do_transform_pose
class Tag_Detector():
    def __init__(self):
        super(Tag_Detector, self).__init__()
        rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.april_tags_callback)
        self.camera_frame = "camera_color_optical_frame"
        self.base_frame = "panda_link0"
        self._tf_listener = tf.TransformListener()

        self.tfBuffer = tf2_ros.Buffer()
        self.transform_listener = tf2_ros.TransformListener(self.tfBuffer)

    # apriltags detection subscriber
    def april_tags_callback(self, data):
        self.detections=data.detections
        # transform it in robot frame
        for tags in self.detections:
            tags.pose.pose.pose = self.transform_in_base(tags.pose.pose.pose)
                
    
    def transform_in_base(self, pose_in_camera):
        try:
            # Get the transform between the camera frame and the robot base frame
            pose_stamp=PoseStamped()
            pose_stamp.pose=pose_in_camera
            transform = self.tfBuffer.lookup_transform(self.base_frame, self.camera_frame, rospy.Time(0),rospy.Duration(2))
            pose_in_base = do_transform_pose(pose_stamp, transform)
            return pose_in_base
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
      
    
    def convert_distribution_to_array(self, use_orientation=False):
        
        self.source_distribution, self.target_distribution= convert_distribution(self.source_distribution, self.target_distribution, use_orientation=use_orientation)
        
    def record_source_distribution(self):
            self.source_distribution=[]
            self.source_distribution=self.continuous_record(self.source_distribution)

    def record_target_distribution(self):
        self.target_distribution=[]
        self.target_distribution=self.continuous_record(self.target_distribution)

    def continuous_record(self, distribution):
        detection_copy=copy(self.detections)
        distribution_copy=copy(distribution)

        if not detection_copy:
            return distribution_copy
        if not distribution_copy:
            distribution=detection_copy
            return detection_copy

        for i in range(len(distribution_copy)):
            if detection_copy and distribution_copy:
                for j in range(len(detection_copy)):
                    if distribution_copy[i].id[0]==detection_copy[j].id[0]:
                        del detection_copy[j]
                        break 
        distribution_copy= distribution_copy + detection_copy 
        return distribution_copy           

    def record_traj_tags(self):
        self.Passive()

        self.end = False
        self.recorded_traj_tag = self.cart_pos.reshape(1,3)
        self.recorded_ori_tag  = self.cart_ori.reshape(1,4)
        while not self.end:

            self.recorded_traj_tag = np.vstack([self.recorded_traj_tag, self.cart_pos])
            self.recorded_ori_tag  = np.vstack([self.recorded_ori_tag,  self.cart_ori])
            self.r_rec.sleep()

    def save_traj_tags(self, data='traj_tags'):

        np.savez(str(pathlib.Path().resolve())+'/data/'+str(data)+'.npz', 
        recorded_traj_tag=self.recorded_traj_tag,
        recorded_ori_tag = self.recorded_ori_tag)

    def load_traj_tags(self, data='traj_tags'):

        data =np.load(str(pathlib.Path().resolve())+'/data/'+str(data)+'.npz')
        self.recorded_traj_tag=data['recorded_traj_tag']
        self.recorded_ori_tag=data['recorded_ori_tag']

    def record_tags(self, distribution):
        start = PoseStamped()

        start.pose.position.x = self.recorded_traj_tag[0,0]
        start.pose.position.y = self.recorded_traj_tag[0,1]
        start.pose.position.z = self.recorded_traj_tag[0,2]

        start.pose.orientation.w = self.recorded_ori_tag[0,0] 
        start.pose.orientation.x = self.recorded_ori_tag[0,1] 
        start.pose.orientation.y = self.recorded_ori_tag[0,2] 
        start.pose.orientation.z = self.recorded_ori_tag[0,3] 
        self.go_to_pose(start)
        j=0
        for i in range(self.recorded_traj_tag.shape[0]-1):
            self.set_attractor(self.recorded_traj_tag[i,:], self.recorded_ori_tag[i,:])
            distribution=self.continuous_record(distribution)
            self.r_rec.sleep() 
        return distribution    

def convert_distribution(source_distribution, target_distribution, use_orientation=False):
    target_array=np.array([], dtype=np.int64).reshape(0,3)
    source_array=np.array([], dtype=np.int64).reshape(0,3)
    for detection_source_in_camera in source_distribution:
        for detection_target_in_camera in target_distribution:
            if detection_source_in_camera.id[0]==detection_target_in_camera.id[0]:  
                detection_target = detection_target_in_camera.pose.pose.pose
                detection_source = detection_source_in_camera.pose.pose.pose
                #Center target
                t=np.array([detection_target.pose.position.x,detection_target.pose.position.y,detection_target.pose.position.z])
                #Center  source
                s=np.array([detection_source.pose.position.x,detection_source.pose.position.y,detection_source.pose.position.z])
                target_array=np.vstack((target_array,t))
                source_array=np.vstack((source_array,s))
                if use_orientation:
                    #Corners source
                    scale_factor=2
                    quat_s=quaternion.from_float_array(np.array([detection_source.pose.orientation.w, detection_source.pose.orientation.x, detection_source.pose.orientation.y, detection_source.pose.orientation.z]))
                    rot_s=quaternion.as_rotation_matrix(quat_s)
                    marker_corners=detect_marker_corners(scale_factor*detection_source_in_camera.size[0])
                    # Rotate marker's corners based on the quaternion
                    rotated_corners = np.dot(rot_s, marker_corners.T).T + s 
                    source_array=np.vstack((source_array,rotated_corners))

                    # Conrners target 
                    quat_t=quaternion.from_float_array(np.array([detection_target.pose.orientation.w, detection_target.pose.orientation.x, detection_target.pose.orientation.y, detection_target.pose.orientation.z]))
                    rot_t=quaternion.as_rotation_matrix(quat_t)
                    marker_corners=detect_marker_corners(scale_factor*detection_target_in_camera.size[0])
                    # Rotate marker's corners based on the quaternion
                    rotated_corners = np.dot(rot_t, marker_corners.T).T + t
                    target_array=np.vstack((target_array,rotated_corners))
    return source_array, target_array


def  detect_marker_corners(marker_dimension):
    marker_corners = np.array([
            [-marker_dimension/2, -marker_dimension/2, 0],
            [-marker_dimension/2, marker_dimension/2, 0],
            [marker_dimension/2, marker_dimension/2, 0],
            [marker_dimension/2, -marker_dimension/2, 0],
             [-marker_dimension/2, -marker_dimension/2, marker_dimension/2],
            [-marker_dimension/2, marker_dimension/2, marker_dimension/2],
            [marker_dimension/2, marker_dimension/2, marker_dimension/2],
            [marker_dimension/2, -marker_dimension/2, marker_dimension/2],
             [-marker_dimension/2, -marker_dimension/2, -marker_dimension/2],
            [-marker_dimension/2, marker_dimension/2, -marker_dimension/2],
            [marker_dimension/2, marker_dimension/2, -marker_dimension/2],
            [marker_dimension/2, -marker_dimension/2, -marker_dimension/2]
            ])    
    return marker_corners
