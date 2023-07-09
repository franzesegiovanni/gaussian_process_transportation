from apriltag_ros.msg import AprilTagDetectionArray
import tf
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
import quaternion
class Tag_Detector():
    def __init__(self):
        super(Tag_Detector, self).__init__()

        rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.april_tags_callback)
        self.camera_frame = "camera_color_optical_frame"
        self.base_frame = "panda_link0"
        self._tf_listener = tf.TransformListener()

        # camera and tags
        self.view_marker = PoseStamped()
        self.view_marker.header.frame_id = "panda_link0"
        self.view_marker.pose.position.x = 0.4585609490798406
        self.view_marker.pose.position.y = -0.04373075604079746
        self.view_marker.pose.position.z = 0.6862181406538658
        self.view_marker.pose.orientation.w = 0.03724672277393113
        self.view_marker.pose.orientation.x =  0.9986700943869168
        self.view_marker.pose.orientation.y =  0.03529063207161849
        self.view_marker.pose.orientation.z = -0.004525063460755314
        self.tfBuffer = tf2_ros.Buffer()
        self.transform_listener = tf2_ros.TransformListener(self.tfBuffer)

    # apriltags detection subscriber
    def april_tags_callback(self, data):
        self.detections=data.detections
    
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
        target_array=np.array([], dtype=np.int64).reshape(0,3)
        source_array=np.array([], dtype=np.int64).reshape(0,3)
        for detection_source_in_camera in self.source_distribution:
            for detection_target_in_camera in self.target_distribution:
                if detection_source_in_camera.id[0]==detection_target_in_camera.id[0]:  
                    detection_target = self.transform_in_base(detection_target_in_camera.pose.pose.pose)
                    detection_source = self.transform_in_base(detection_source_in_camera.pose.pose.pose)
                    #Center target
                    t=np.array([detection_target.pose.position.x,detection_target.pose.position.y,detection_target.pose.position.z])
                    #Center  source
                    s=np.array([detection_source.pose.position.x,detection_source.pose.position.y,detection_source.pose.position.z])
                    target_array=np.vstack((target_array,t))
                    source_array=np.vstack((source_array,s))
                    if use_orientation==True:
                        # Conrners target 
                        quat_t=quaternion.from_float_array(np.array([detection_target.pose.orientation.w, detection_target.pose.orientation.x, detection_target.pose.orientation.y, detection_target.pose.orientation.z]))
                        rot_t=quaternion.as_rotation_matrix(quat_t)
                        marker_corners=marker_corners(self.marker_dimensions)
                        translated_corners = marker_corners + t
                        # Rotate marker's corners based on the quaternion
                        rotated_corners = np.dot(rot_t, translated_corners.T).T
                        target_array=np.vstack((target_array,rotated_corners))

                        #Corners source

                        quat_s=quaternion.from_float_array(np.array([detection_source.pose.orientation.w, detection_source.pose.orientation.x, detection_source.pose.orientation.y, detection_source.pose.orientation.z]))
                        rot_s=quaternion.as_rotation_matrix(quat_s)
                        marker_corners=marker_corners(self.marker_dimensions)
                        translated_corners = marker_corners + s
                        # Rotate marker's corners based on the quaternion
                        rotated_corners = np.dot(rot_s, translated_corners.T).T
                        source_array=np.vstack((target_array,rotated_corners))
                        
        self.target_distribution=target_array
        self.source_distribution=source_array
        
    def record_source_distribution(self):
        self.source_distribution=self.detections

    def record_target_distribution(self):
        self.target_distribution=self.detections    

def  marker_corners(marker_dimension):
    marker_corners = np.array([
            [-marker_dimension/2, -marker_dimension/2, 0],
            [-marker_dimension/2, marker_dimension/2, 0],
            [marker_dimension/2, marker_dimension/2, 0],
            [marker_dimension/2, -marker_dimension/2, 0]])    
    return marker_corners