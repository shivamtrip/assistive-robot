#!/usr/local/lib/graspnet_env/bin/python3

import rospy
import sys
import os
import cv2
from sensor_msgs.msg import PointCloud,PointCloud2, PointField
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
import json
from rospy.numpy_msg import numpy_msg
from grasp_detector.msg import GraspPoseAction, GraspPoseResult
# from utils import centroid_grasp
from graspgen import GraspGenerator
import actionlib
import tf
import tf.transformations
from tf import TransformerROS
from tf import TransformListener, TransformBroadcaster
from yolo.msg import Detections
import struct
import open3d as o3d

class GraspDetector():

    def __init__(self):
        rospy.init_node('grasp_detector_node', anonymous=True)

        self._as = actionlib.SimpleActionServer('grasp_detector', GraspPoseAction, execute_cb=self.inference, auto_start = False)
        self.bridge = CvBridge()
        
        self.image_scale = 1/rospy.get_param('/image_shrink/downscale_ratio') #rescaling bbox detections on low res image to high res
        self.transform_source_frame = rospy.get_param('/graspnet/tf_source_frame')
        self.transform_target_frame = rospy.get_param('/graspnet/tf_target_frame')
        self.publish_frame_name = rospy.get_param('/graspnet/tf_publish_frame_name')

        self.tf_publisher = TransformBroadcaster()
        
        self.mode = rospy.get_param('/graspnet/grasp_mode')

        rospy.loginfo(f"[{rospy.get_name()}] " + "Loading model")

        self.grasp_obj = GraspGenerator(
            checkpoint_path =   rospy.get_param('/graspnet/checkpoint_path'), 
            num_point =         rospy.get_param('/graspnet/num_point'), 
            num_view =          rospy.get_param('/graspnet/num_view'),
            collision_thresh =  rospy.get_param('/graspnet/collision_thresh'), 
            voxel_size =        rospy.get_param('/graspnet/voxel_size'),
            max_depth =         rospy.get_param('/graspnet/max_depth'),
            num_angle =         rospy.get_param('/graspnet/num_angle'),
            num_depth =         rospy.get_param('/graspnet/num_depth'),
            cylinder_radius =   rospy.get_param('/graspnet/cylinder_radius'),
            input_feature_dim = rospy.get_param('/graspnet/input_feature_dim'),
            hmin =              rospy.get_param('/graspnet/hmin'),
            hmax_list =         rospy.get_param('/graspnet/hmax_list'),
            device_name=        rospy.get_param('/graspnet/model_device_name'),
        )

        self._as.start()
        rospy.loginfo(f"[{rospy.get_name()}] " + " Grasp Pose Server Started")

    def getImages(self, color, depth):
        self.depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        self.color_img = self.bridge.imgmsg_to_cv2(color, desired_encoding="bgr8")
        self.obtainedInitialImages = True
        
    def getBboxes(self,msg):
        
        num_detections = (msg.nPredictions)
        self.bboxes = {
            "boxes" : np.array(msg.box_bounding_boxes).reshape(num_detections,4),
            "box_classes" : np.array(msg.box_classes).reshape(num_detections)
        }

    def publish_transform(self, tf_matrix):
        quat = tf.transformations.quaternion_from_matrix(tf_matrix)
        self.tf_publisher.sendTransform(
            (tf_matrix[0][3],tf_matrix[1][3],tf_matrix[2][3]),
            quat,
            rospy.Time.now(),
            self.publish_frame_name,
            self.transform_target_frame
        )

        angles = tf.transformations.euler_from_quaternion(quat)

        angles = np.array(angles)
        angles[0] = 0
        angles[1] = 0

        roll, pitch, yaw = angles[0],angles[1],angles[2]

        #ignores pitch and roll.
        quat_yaw_only = tf.transformations.quaternion_from_euler(roll,pitch,yaw)

        #publishes the grasp pose with only yaw angle
        self.tf_publisher.sendTransform(
            (tf_matrix[0][3], tf_matrix[1][3], tf_matrix[2][3]),
            quat_yaw_only,
            rospy.Time.now(),
            self.publish_frame_name + '_yaw',
            self.transform_target_frame
        )


    def runGraspnet(self, color, depth, bbox, intrinsics, vis_flag = False):
        '''
            runs inference on images and return grasp poses 
        '''
        
        grasp_obj = self.grasp_obj
        end_points, cloud = grasp_obj.get_and_process_data(color, depth, bbox, intrinsics)
        gg = grasp_obj.get_grasps(grasp_obj.net, end_points)

        if grasp_obj.collision_thresh> 0:
            gg = grasp_obj.collision_detection(gg, np.array(cloud.points))
        grasp_obj.sort_grasps(gg)
        
        if(vis_flag):
            grasp_obj.vis_grasps(gg, cloud)

        return gg

    def inference(self, goal):
        
        rgb_image = self.bridge.imgmsg_to_cv2(goal.rgb_image, desired_encoding="passthrough")
        depth_image = self.bridge.imgmsg_to_cv2(goal.depth_image, desired_encoding="passthrough")
        box = np.array(goal.bbox)
        extrinsics = np.array(goal.extrinsics).reshape(4,4)
        intrinsics = np.array(goal.intrinsics).reshape(3,3)

        grasp_response_msg = GraspPoseResult()

        if self.mode == 'graspnet':
            rospy.loginfo(f"[{rospy.get_name()}] " + "Running GraspNet")
            graspObj = self.runGraspnet(
                color = rgb_image,
                depth = depth_image, 
                bbox = box,
                intrinsics = intrinsics,
                vis_flag = False
            )
            if len(graspObj) == 0:
                grasp_response_msg.success = False
                self._as.set_aborted(grasp_response_msg)
                rospy.loginfo(f"[{rospy.get_name()}] " + "Graspnet failed to find a grasp. Falling back to centroid")
            else:
                rospy.loginfo(f"[{rospy.get_name()}] " + "Found grasp from network!")
                rot_mat_grasp = graspObj[0].rotation_matrix.astype(np.float32)
                t_grasp = graspObj[0].translation.astype(np.float32)

                tf_mat_cam2obj = np.hstack((rot_mat_grasp,t_grasp.reshape((3,1))))
                tf_mat_cam2obj = np.vstack((tf_mat_cam2obj, np.array([[0,0,0,1]])))

                tf_base2obj = np.matmul(extrinsics,tf_mat_cam2obj)
                angles = tf.transformations.euler_from_matrix(tf_base2obj)

                grasp_response_msg.x = tf_base2obj[0][3]
                grasp_response_msg.y = tf_base2obj[1][3]
                grasp_response_msg.z = tf_base2obj[2][3]
                grasp_response_msg.yaw = angles[2] ## TODO : CHECK IF CORRECT.
                
                self.publish_transform(tf_base2obj)
                
                grasp_response_msg.success = True
                self._as.set_succeeded(grasp_response_msg)

        # if graspNetFailed or self.mode == 'centroid':
        #     rospy.loginfo(f"[{rospy.get_name()}] " + "Running Centroid")
        #     graspObj_centroid = self.grasp_obj.centroid_grasp(rgb_image, depth_image, mask, self.intrinsics)
        #     graspObj_centroid = np.vstack((graspObj_centroid.reshape(3,1),np.array([[1]])))
        #     tf_base2obj = np.matmul(extrinsics, graspObj_centroid)
        #     graspPose.x = tf_base2obj[0][0]
        #     graspPose.y = tf_base2obj[1][0]
        #     graspPose.z = tf_base2obj[2][0]
        #     graspPose.yaw = 0
        #     mat = np.eye(4)
        #     mat[:3, 3] = tf_base2obj[0:3].flatten()
        #     self.publish_transform(mat)


        





        
if __name__ == '__main__':
    grasp_gen = GraspDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo(f"[{rospy.get_name()}] " + "Shutting down")



    # g = grasp()
    # g.object_id = gg[0].object_id
    # g.score = gg[0].score
    # g.width = gg[0].width
    # g.height = gg[0].height
    # g.depth = gg[0].depth
    # g.translation = gg[0].translation.astype(np.float32)
    # g.rotation = np.ravel(gg[0].rotation_matrix.astype(np.float32))
    # to enable this, add to cmakelists again.