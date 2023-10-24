import numpy as np
import quaternion
def convert_distribution(source_distribution, target_distribution, use_orientation=False):
    target_array=np.array([], dtype=np.int64).reshape(0,3)
    source_array=np.array([], dtype=np.int64).reshape(0,3)
    for detection_source_in_camera in source_distribution:
        for detection_target_in_camera in target_distribution:
            if detection_source_in_camera.id[0]==detection_target_in_camera.id[0]:  
                detection_target = detection_target_in_camera.pose.pose.pose
                detection_source = detection_source_in_camera.pose.pose.pose
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
                    # rotated_corners[:,-1]=s[-1]
                    source_array=np.vstack((source_array,rotated_corners))

                    # Conrners target 
                    quat_t=quaternion.from_float_array(np.array([detection_target.pose.orientation.w, detection_target.pose.orientation.x, detection_target.pose.orientation.y, detection_target.pose.orientation.z]))
                    rot_t=quaternion.as_rotation_matrix(quat_t)
                    marker_corners=detect_marker_corners(scale_factor*detection_target_in_camera.size[0])
                    # Rotate marker's corners based on the quaternion
                    rotated_corners = np.dot(rot_t, marker_corners.T).T + t
                    # rotated_corners[:,-1]=t[-1]
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