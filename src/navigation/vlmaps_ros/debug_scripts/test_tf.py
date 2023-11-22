#!~/miniconda3/envs/vlmaps/bin/python

""" Tests whether all the required transforms are available or not """

import sys
sys.path.append('/home/abhinav/alfred-autonomy/src/navigation/vlmaps_ros')

import rospy
import sys
import os
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
import json
from rospy.numpy_msg import numpy_msg
import actionlib
import tf
import argparse
import tf2_ros
import tf.transformations
from tf import TransformerROS
from tf import TransformListener, TransformBroadcaster
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, TransformStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import Odometry


class RecData(object):

    def __init__(self,params:dict):

        print(" Converting data .............")
        self.params = params
        self.is_rtabmap = params['is_rtabmap']
        self.fps = params["fps"]
        self.frame_id = 0
        self.bridge = CvBridge()
        self.tf_ros = TransformerROS()
        self.tf_listener = TransformListener()
        self.tf_publisher = TransformBroadcaster()
        self.img_save_dir = params["img_save_dir"]
        self.transform_target_frame = "base_link"
        self.publish_frame_name = "camera_color_optical_frame"
       
        if not os.path.isdir(self.img_save_dir):
            raise Exception("img_save_dir is not a directory")
        
        self.clean_dir(self.img_save_dir)

        self.pose_dir = os.path.join(self.img_save_dir, "pose")
        self.depth_dir = os.path.join(self.img_save_dir, "depth")
        self.rgb_dir = os.path.join(self.img_save_dir, "rgb")
        self.semantic_dir = os.path.join(self.img_save_dir, "semantic") # stores placeholder dummy data
        self.add_params_dir = os.path.join(self.img_save_dir, "add_params")

        dir_list = [self.pose_dir, self.depth_dir, self.rgb_dir,self.semantic_dir, self.add_params_dir]
        self.create_dir(dir_list)

        self.cameraInfoSub = rospy.Subscriber(params['intrinsic_topic'], CameraInfo, self.getCamInfo)
        self.color_sub = message_filters.Subscriber(params['color_topic'], Image)
        self.depth_sub = message_filters.Subscriber(params['depth_topic'], Image)

        # Change subscriber based on whether rtabmap is used or not
        if(self.params['is_rtabmap']):
            self.pose_sub = message_filters.Subscriber(params['pose_topic'], Odometry)
        else:
            self.pose_sub = message_filters.Subscriber(params['pose_topic'], PoseWithCovarianceStamped)

        self.sensor_sub_timesynced = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub,self.pose_sub], 10, 0.1, allow_headerless=True)
        # self.tf_buffer = tf2_ros.Buffer()
        self.tf_ros = TransformerROS()
        self.tf_listener = TransformListener()
        self.tf_publisher = TransformBroadcaster()

        self.last_time = rospy.Time.now()
        self.current_time = None

        # register callback
        self.sensor_sub_timesynced.registerCallback(self.sub_callback)

    def create_dir(self,dir_list:list):
        """ Creates the directory structure required by VLMaps """

        for dir in dir_list:
            if not os.path.isdir(dir):
                os.mkdir(dir)
    
    def clean_dir(self,dir:str):
        """ Cleans the directory structure required by VLMaps """

        # assert that directory path ends with 'rosbag_data'
        if not dir.endswith('rosbag_data'):
            raise Exception("Directory path should end with 'rosbag_data'")

        if os.path.isdir(dir):
            os.system(f"rm -rf {dir}/*")

    def getTransform(self, target_frame, source_frame):

        # Get the transform between the two frames
        tf_matrix = None
        target_frame = 'base_link'
        source_frame = 'camera_color_optical_frame'

        # list all frames in the tf
        # all_frames = self.tf_listener.getFrameStrings()
        # print("all_frames=",all_frames)

        if(self.tf_listener.frameExists(target_frame)):
           rospy.loginfo(f"[{rospy.get_name()}] " + "Frame {} exists".format(target_frame))

        if(self.tf_listener.frameExists(source_frame)):
              rospy.loginfo(f"[{rospy.get_name()}] " + "Frame {} exists".format(source_frame))

        if self.tf_listener.frameExists(target_frame) and self.tf_listener.frameExists(source_frame):

            # self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
            position, quaternion = self.tf_listener.lookupTransform(target_frame, source_frame,rospy.Time()) 
            tf_matrix = self.tf_ros.fromTranslationRotation(position,quaternion)

            if(tf_matrix is None):
                rospy.logwarn(f"[{rospy.get_name()}] " + "Transform not found between {} and {}".format(target_frame, source_frame))
                rospy.sleep(0.02)
            else:
                return tf_matrix
            
        else:
            rospy.logwarn(f"[{rospy.get_name()}] " + "Frames not found between {} and {}".format(target_frame, source_frame))
            rospy.sleep(0.02)

    def getCamInfo(self,msg):
        """ Subscribe to camera info and store the intrinsics (just once)"""
        self.intrinsics = np.array(msg.K).reshape(3,3)
        
        rospy.logdebug(f"[{rospy.get_name()}] " + "Obtained camera intrinsics")

        # Read the transform from "base_link" to "camera_color_optical_frame"
        # handle exception and only call when transform is available
        self.tf_base2cam = np.array([[-0.02758661, -0.05052392,  0.99834178,  0.03980751],
                                    [ 0.04150318,  0.9978028,   0.05164348, -0.03182379],
                                    [-0.99875746,  0.04285903, -0.02542909,  1.30937385],
                                    [ 0.,          0.,          0.,          1.        ]])
        
        tf_ = self.getTransform(self.transform_target_frame, self.publish_frame_name)
        if(tf_ is not None):
            self.tf_base2cam = tf_
            rospy.logdebug(f"[{rospy.get_name()}] " + "Obtained tf_base2cam transform")
        else:
            euler = tf.transformations.euler_from_matrix(self.tf_base2cam)
            print("euler angles=",euler)
            rospy.logwarn(f"[{rospy.get_name()}] " + "Using default tf value {}".format(self.tf_base2cam))
        
        self.obtainedIntrinsics = True
        self.cameraInfoSub.unregister()

    def get_rtabmap_pose(self, odom_msg):

        # Convert odom_msg to pose_msg
        pose_msg = PoseWithCovarianceStamped()
        pose_msg = odom_msg

        # Get the pose
        pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z])
        rot = np.array([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])

        return pos, rot
    
    def get_amcl_pose(self, pose_msg):

        # Get the pose
        pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z])
        rot = np.array([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])

        return pos, rot

    def sub_callback(self, color_msg, depth_msg, odom_msg):
        """ Callback function to save the data """
    
        # Subscribe only at desired frequency
        self.current_time = rospy.Time.now()
        if(self.current_time - self.last_time < rospy.Duration.from_sec(1.0/self.fps)):
            return
        
        if(self.is_rtabmap):
            pos, rot = self.get_rtabmap_pose(odom_msg)
        else:
            pos, rot = self.get_amcl_pose(odom_msg)
        
        
        tf_matrix = self.tf_ros.fromTranslationRotation(pos,rot)

        pos = tf_matrix[:3,3]
        rot = tf.transformations.quaternion_from_matrix(tf_matrix) # (x,y,z,w)

        # Get the rgb image
        rgb_img = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        # rotate by 90
        rgb_img = np.rot90(rgb_img, k=3)
      
        # Get the depth image
        depth_img = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width, -1)
        depth_img = depth_img.squeeze().astype(np.float32)
        # rotate by 90
        depth_img = np.rot90(depth_img, k=3)

        # Save the data
        self.save(rgb_img, depth_img, pos, rot)

        # Update the last time
        self.last_time = self.current_time

    def save(self, rgb_img, depth_img, pos, rot):
        """ Save the data to the directory structure required by VLMaps """

        pass

if(__name__=='__main__'):

    rospy.init_node('VLMapDataRecorder', anonymous=True)

    # Take fps argumnent
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--img_save_dir", type=str, default='/home/abhinav/alfred-autonomy/src/navigation/vlmaps_ros/data/rosbag_data')
    parser.add_argument("--color_topic", type=str, default='/camera/color/image_raw')
    parser.add_argument("--depth_topic", type=str, default='/camera/aligned_depth_to_color/image_raw')
    parser.add_argument("--pose_topic", type=str, default='/amcl_pose')
    parser.add_argument("--intrinsic_topic", type=str, default='/camera/color/camera_info')
    parser.add_argument("--is_rtabmap", type=bool, default=False)

    args = parser.parse_args()
    params = vars(args)

    if(params['is_rtabmap']):
        params['pose_topic'] = '/rtabmap/odom'

    else:
        params['pose_topic'] = '/amcl_pose'

    rospy.loginfo(f"[{rospy.get_name()}] " + "Recording data FPS: {}".format(params["fps"]))

    # check if dir exists
    if not os.path.isdir(params["img_save_dir"]):
        os.mkdir(params['img_save_dir'])

    rec_data_obj = RecData(params=params)
    try:
        rospy.spin()

    except KeyboardInterrupt:
        rospy.loginfo(f"[{rospy.get_name()}] " + "Shutting down")