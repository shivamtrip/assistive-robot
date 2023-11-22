#!/usr/bin/env python3

import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import tf.transformations
from tf import TransformerROS
from tf import TransformListener, TransformBroadcaster

# listen to tf between /base_link and /camera_color_optical_frame
class PublishTF:

    def __init__(self) -> None:

        rospy.init_node('publish_tf', anonymous=True)

        self.tfros = TransformerROS()
        self.tf_listener = tf.TransformListener()
        self.tf_publisher = tf.TransformBroadcaster()
        
        self.transform_target_frame = "base_link"
        self.publish_frame_name = "camera_color_optical_frame"

        # self.transform_target_frame = "odom"
        # self.publish_frame_name = "base_link"

        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)

        # register callback
        color_sub.registerCallback(self.sub_callback)
        
    def sub_callback(self, color_msg):
        """ Callback function to save the data """
        # Get the transform between the two frames
        tf_matrix = self.getTransform(self.transform_target_frame, self.publish_frame_name)
        print("tf_matrix: ", tf_matrix)
        if(tf_matrix is None):
            print("No tf matrix")
        
    def getTransform(self, target_frame, source_frame):

        # Get the transform between the two frames
        tf_matrix = None
        if True:
            t = self.tf_listener.getLatestCommonTime(target_frame, source_frame)
            position, quaternion = self.tf_listener.lookupTransform(target_frame, source_frame, t)
            tf_matrix = self.tfros.fromTranslationRotation(position,quaternion)

            if(tf_matrix is None):
                rospy.logerr(f"[{rospy.get_name()}] " + "Transform not found between {} and {}".format(target_frame, source_frame))
                return None

            return tf_matrix


if __name__ == "__main__":
    
    node = PublishTF()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass