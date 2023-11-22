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
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


class PublishPCD:

        
    def callback(self, rgb_msg, depth_msg):
        rgb_image = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1)
        depth_image = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width, -1)

        print("depth shape: ", depth_image.shape)
        print("rgb shape: ", rgb_image.shape)
        h,w = rgb_image.shape[:2]

        # resize depth to (640, 480)
        # depth_image = cv2.resize(depth_image, (rgb_image.shape[0],rgb_image.shape[1]), interpolation=cv2.INTER_NEAREST)
        depth = np.array(depth_image, dtype=np.float32)/1000.0
        depth = np.squeeze(depth)
        depth= np.rot90(depth, k=3)

        # Convert intrinsics to account for 90 deg rotation
        intrinsics = self.convert_intrinsics(self.intrinsics,rgb_image.shape[0],rgb_image.shape[1])

        pcd,_ = self.depth2pc_real(depth, intrinsics)

        depth_filter = pcd[2, :] < 5.0
        pcd = pcd[:, depth_filter]

        # Swap x and y values since the pcd is made from 90 deg rotated image
        tf_camera_optical2pcd = self.get_tf_camera_optical2pcd()
        # transform pcd to camera_optical frame
        pcd = tf_camera_optical2pcd @ np.vstack([pcd, np.ones((1, pcd.shape[1]))])
        pcd = pcd[:3, :]

        print("intrinsics: ", intrinsics)
        assert pcd.shape[0] == 3, "point cloud shape is not 3xN"

        print(np.min(pcd), np.max(pcd))

        # Create a PointCloud2 message
        header = rgb_msg.header
        header.stamp = rgb_msg.header.stamp
        header.frame_id = 'camera_color_optical_frame'  # Set the frame ID
        # add image colors to it
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        # rgb_image = rgb_image.reshape((rgb_image.shape[0]*rgb_image.shape[1], 3))
        # rgb_image = np.array(rgb_image, dtype=np.uint8)
        pc_msg = pc2.create_cloud_xyz32(header, np.transpose(pcd))

        self.pub_pcd.publish(pc_msg)

    def get_tf_camera_optical2pcd(self,):
       
        """ Return the transformation from the camera optical frame to the local pcd frame
        """
        tf = np.eye(4)
        # rotate clockwise 90 deg around z
        rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        tf[:3, :3] = rot
        return tf
    
    def convert_intrinsics(self,intrinsics:np.ndarray,h,w):

        assert h<w,"h can't be larger than w"
        fx_old = intrinsics[0,0]
        fy_old = intrinsics[1,1]
        cx_old = intrinsics[0,2]
        cy_old = intrinsics[1,2]

        fx_new = fy_old
        fy_new = fx_old
        cx_new = h - cy_old
        cy_new = cx_old

        rotated_intrinsic = np.array([[fx_new,0,cx_new],[0,fy_new,cy_new],[0,0,1]])

        return rotated_intrinsic

    def depth2pc_real(self,depth,K):
        """
        Return 3xN array
        """

        while(K is None):
            print("No intrinsics provided")
            rospy.sleep(1)

        h, w = depth.shape
        K_inv = np.linalg.inv(K)
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        x = x.reshape((1, -1))[:, :]
        y = y.reshape((1, -1))[:, :]
        z = depth.reshape((1, -1))[:, :]

        # print max and min of depth
        print("max depth: ", np.max(z))
        print("min depth: ", np.min(z))

        p_2d = np.vstack([x, y, np.ones_like(x)])
        pc = K_inv @ p_2d
        pc = pc * z
        mask = pc[2, :] > 0.1
        print("PC shape: ", pc.shape)
        return pc, mask
        

    def getCamInfo(self,msg):
        """ Subscribe to camera info and store the intrinsics (just once)"""
        self.intrinsics = np.array(msg.K).reshape(3,3)
        self.obtainedIntrinsics = True
        rospy.logdebug(f"[{rospy.get_name()}] " + "Obtained camera intrinsics")
        # Write camera intrinsics to file
        self.cameraInfoSub.unregister()
            

    def main(self):
        rospy.init_node('publish_pcd', anonymous=False)

        # initialize intrinsics
        # self.intrinsics = np.eye(3)
        # self.intrinsics[0, 0] = self.intrinsics[1, 1] = 605.6397094726562
        # self.intrinsics[0, 2] = 325.9149475097656
        # self.intrinsics[1, 2] = 245.17396545410156

        self.intrinsics= None

        ## Real world ros bag params
        # self.cameraInfoSub = rospy.Subscriber('camera/color/camera_info', CameraInfo, self.getCamInfo)
        # self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        # self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        # self.sensor_sub_timesynced = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1, allow_headerless=True)

        # Simulation ros bag params
        self.cameraInfoSub = rospy.Subscriber('camera/color/camera_info', CameraInfo, self.getCamInfo)
        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
        self.sensor_sub_timesynced = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1, allow_headerless=True)

        # initialize point cloud publisher that publish point clouds
        self.pub_pcd = rospy.Publisher('point_cloud_topic', PointCloud2, queue_size=10)


        # register callback
        self.sensor_sub_timesynced.registerCallback(self.callback)

        rospy.loginfo(f"[{rospy.get_name()}] " + "Node Ready")

if __name__ == "__main__":
    
    node = PublishPCD()
    node.main()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass