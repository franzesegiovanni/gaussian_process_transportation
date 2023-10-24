import pathlib
import tf
import tf2_ros
import rospy
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import do_transform_pose
import pickle
from panda_ros import Panda # install from https://github.com/platonics-delft/panda-ros-py/tree/main
import os
# this file requires that you launch the panda controller, real or simulated and the statis transformation of the camera optical frame and the panda hand (or any other frame of the robot). This script was created because the original data was recorded in the camera frame and we want to convert it to the base frame of the robot.
class Converter():
    def __init__(self):
        rospy.init_node('converter', anonymous=True)
        self.camera_frame = "camera_color_optical_frame"
        self.base_frame = "panda_link0"
        self._tf_listener = tf.TransformListener()

        # camera and tags
        self.view_marker = PoseStamped()
        self.view_marker.header.frame_id = "panda_link0"
        self.view_marker.pose.position.x = 0.08143687722855915
        self.view_marker.pose.position.y = 0.31402779786395074
        self.view_marker.pose.position.z = 0.8247450387759941
        self.view_marker.pose.orientation.w =0.1649396209403439
        self.view_marker.pose.orientation.x =  0.876597038344831
        self.view_marker.pose.orientation.y =  -0.22170860567236517
        self.view_marker.pose.orientation.z =  0.39397046435209987
        self.tfBuffer = tf2_ros.Buffer()
        self.transform_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.panda=Panda()
        rospy.sleep(1)
    def go_to_recording_pose(self):
        self.panda.go_to_pose(self.view_marker)

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

    def load_distribution(self, filename):
        self.filename = filename
        with open(filename,"rb") as file:
            self.detections = pickle.load(file)
        for tags in self.detections:
            tags.pose.pose.pose = self.transform_in_base(tags.pose.pose.pose)


    def save_in_base_frame(self, path):
        new_filename = path+ pathlib.Path(self.filename).stem + ".pkl"
        with open(new_filename,"wb") as file:
            pickle.dump(self.detections,file)   


if __name__=='__main__':
    converter=Converter()
    converter.go_to_recording_pose()
    file_dir= os.path.dirname(__file__)
    data_folder =file_dir+ "/results/dressing/camera_frame/"
    for filename in sorted(os.listdir(data_folder)):
        if filename.endswith(".pkl") and not "demo" in filename:
            if "target" in filename or "source" in filename:
                print(filename)
                target_dist= converter.load_distribution(data_folder+filename)
                converter.save_in_base_frame(file_dir+ "/results/dressing/")        


        