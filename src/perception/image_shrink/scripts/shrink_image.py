#!/usr/bin/env python3

import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from rospy.numpy_msg import numpy_msg
import tf


class ShrinkImage:

        
    def callback(self, ros_rgb_image):
        rgb_image = self.cv_bridge.imgmsg_to_cv2(ros_rgb_image, 'bgr8')
        rgb_image = cv2.resize(rgb_image, (0, 0), fx = self.scale, fy = self.scale) # 1/4th of the original size.
        imgmsg = self.cv_bridge.cv2_to_imgmsg(rgb_image)
        imgmsg.header.stamp = ros_rgb_image.header.stamp
        self.pub.publish(imgmsg)
            

    def main(self):
        rospy.init_node('shrink_image', anonymous=False)

        self.scale = rospy.get_param('/image_shrink/downscale_ratio')
        self.rgb_image_subscriber = rospy.Subscriber(rospy.get_param('/image_shrink/subscribe_topic'), Image,self.callback )
        self.pub = rospy.Publisher(rospy.get_param('/image_shrink/publish_topic'), Image, queue_size=1)
        
        self.cv_bridge = CvBridge()

        rospy.loginfo(f"[{rospy.get_name()}] " + "Node Ready")

    
if __name__ == "__main__":
    
    node = ShrinkImage()
    node.main()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass