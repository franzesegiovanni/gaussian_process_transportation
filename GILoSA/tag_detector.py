from apriltag_ros.msg import AprilTagDetectionArray
import tf
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
import quaternion
from copy import copy
import pathlib
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
        target_array=np.array([], dtype=np.int64).reshape(0,3)
        source_array=np.array([], dtype=np.int64).reshape(0,3)
        for detection_source_in_camera in self.source_distribution:
            for detection_target_in_camera in self.target_distribution:
                if detection_source_in_camera.id[0]==detection_target_in_camera.id[0]:  
                    detection_target = detection_target_in_camera.pose.pose.pose
                    detection_source = detection_source_in_camera.pose.pose.pose
                    #Center target
                    t=np.array([detection_target.pose.position.x,detection_target.pose.position.y,detection_target.pose.position.z])
                    #Center  source
                    s=np.array([detection_source.pose.position.x,detection_source.pose.position.y,detection_source.pose.position.z])
                    target_array=np.vstack((target_array,t))
                    source_array=np.vstack((source_array,s))
                    if use_orientation==True:
                        #Corners source

                        quat_s=quaternion.from_float_array(np.array([detection_source.pose.orientation.w, detection_source.pose.orientation.x, detection_source.pose.orientation.y, detection_source.pose.orientation.z]))
                        rot_s=quaternion.as_rotation_matrix(quat_s)
                        marker_corners=detect_marker_corners(1*detection_source_in_camera.size[0])
                        # Rotate marker's corners based on the quaternion
                        rotated_corners = np.dot(rot_s, marker_corners.T).T + s 
                        rotated_corners[:,-1]=s[-1]
                        source_array=np.vstack((source_array,rotated_corners))

                        # Conrners target 
                        quat_t=quaternion.from_float_array(np.array([detection_target.pose.orientation.w, detection_target.pose.orientation.x, detection_target.pose.orientation.y, detection_target.pose.orientation.z]))
                        rot_t=quaternion.as_rotation_matrix(quat_t)
                        marker_corners=detect_marker_corners(1*detection_target_in_camera.size[0])
                        # Rotate marker's corners based on the quaternion
                        rotated_corners = np.dot(rot_t, marker_corners.T).T + t
                        rotated_corners[:,-1]=t[-1]
                        target_array=np.vstack((target_array,rotated_corners))

  
                        
        self.target_distribution=target_array
        self.source_distribution=source_array
        
    # def record_source_distribution(self):
    #     self.source_distribution=self.detections

    # def record_target_distribution(self):
    #     self.target_distribution=self.detections

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
            # print("No detection")
            return distribution_copy
        if not distribution_copy:
            distribution=detection_copy
            # print(detection_copy)
            # print("No distribution")
            return detection_copy

        for i in range(len(distribution_copy)):
            if detection_copy and distribution_copy:
                for j in range(len(detection_copy)):
                    # print(distribution_copy[i].id[0])
                    # print(detection_copy[j].id[0])
                    if distribution_copy[i].id[0]==detection_copy[j].id[0]:
                        distribution_copy[i]=copy(detection_copy[j])
                        del detection_copy[j]
                        break 
        distribution_copy= distribution_copy + detection_copy 
        return distribution_copy           

    def Record_traj_tags(self):
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

    def Record_tags(self, distribution):
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
            if i>20 and i < self.recorded_traj_tag.shape[0]-20 and np.linalg.norm(self.recorded_traj_tag[i-20,:]-self.recorded_traj_tag[i+20,:])<0.0005 and j>20:
                print("Saving frames")
                distribution=self.continuous_record(distribution)
                # print(distribution)
                j=0
            j=j+1
            self.r_rec.sleep() 
        return distribution    

    def Record_tags_goto(self, distribution):
        start = PoseStamped()
        start.pose.position.x = 0.40375158
        start.pose.position.y = -0.01048578
        start.pose.position.z = 0.75245844
        start.pose.orientation.w =  0.08399367
        start.pose.orientation.x = 0.68521328
        start.pose.orientation.y = 0.7230002
        start.pose.orientation.z = -0.02625647
        self.go_to_pose(start)
        rospy.sleep(2)
        distribution=self.continuous_record(distribution)
        rospy.sleep(2)
        # [ 0.40375158, -0.01048578,  0.75245844]
        # [ 0.08399367,  0.68521328,  0.7230002 , -0.02625647]
        start.pose.position.x = 0.42474412
        start.pose.position.y = -0.00146312
        start.pose.position.z = 0.76038187
        start.pose.orientation.w =  0.31872356
        start.pose.orientation.x = 0.63964127
        start.pose.orientation.y = 0.64930498
        start.pose.orientation.z = -0.26013055
        # [ 0.42474412, -0.00146312,  0.76038187]
        # [ 0.31872356,  0.63964127,  0.64930498, -0.26013055]
        self.go_to_pose(start)
        rospy.sleep(2)
        distribution=self.continuous_record(distribution)
        rospy.sleep(2)
        start.pose.position.x = 0.4272796
        start.pose.position.y =-0.0647395
        start.pose.position.z = 0.8631258
        start.pose.orientation.w =  0.51575005
        start.pose.orientation.x = 0.50598534
        start.pose.orientation.y =  0.48100053
        start.pose.orientation.z = -0.49659837
        rospy.sleep(2)
        self.go_to_pose(start)
        rospy.sleep(2)
        distribution=self.continuous_record(distribution)
        #[ 0.4272796, -0.0647395,  0.8631258]
        #[ 0.51575005,  0.50598534,  0.48100053, -0.49659837]
        return distribution


def  detect_marker_corners(marker_dimension):
    marker_corners = np.array([
            [-marker_dimension/2, -marker_dimension/2, 0],
            [-marker_dimension/2, marker_dimension/2, 0],
            [marker_dimension/2, marker_dimension/2, 0],
            [marker_dimension/2, -marker_dimension/2, 0]])    
    return marker_corners