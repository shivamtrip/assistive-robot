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
        
        # while True:
        #     try:
        #         val = rospy.get_param('/image_shrink/downscale_ratio')
        #     except KeyError as k:
        #         rospy.loginfo(f"[{rospy.get_name()}] " + "Waiting for image_shrink node to start")
        #         rospy.sleep(1)
        #         continue
        #     break

        self.mode = rospy.get_param('/graspnet/grasp_mode')
        intrinsic_topic = rospy.get_param('/graspnet/camera_info_topic')
        depth_topic = rospy.get_param('/graspnet/depth_image_topic')
        color_topic = rospy.get_param('/graspnet/color_image_topic')
        
        self.objectOfInterest = 64
        
        self.color_sub = message_filters.Subscriber(color_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.box_sub = rospy.Subscriber("object_bounding_boxes", Detections,self.getBboxes)
        self.cameraInfoSub = rospy.Subscriber(intrinsic_topic, CameraInfo, self.getCamInfo)
        self.image_scale = 1/rospy.get_param('/image_shrink/downscale_ratio') #rescaling bbox detections on low res image to high res
        print(self.image_scale)
        self.depth_img = None
        self.color_img = None
        self.image_sub_timesynced = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1, allow_headerless=True)

        # transform frames
        self.transform_source_frame = rospy.get_param('/graspnet/tf_source_frame')
        self.transform_target_frame = rospy.get_param('/graspnet/tf_target_frame')
        self.publish_frame_name = rospy.get_param('/graspnet/tf_publish_frame_name')
        self.tf_ros = TransformerROS()
        self.tf_listener = TransformListener()
        self.tf_publisher = TransformBroadcaster()

        self.image_sub_timesynced.registerCallback(self.getImages)
        self.temppub = rospy.Publisher("pointcloud", PointCloud2, queue_size=1)

        self.bboxes = None
        self.obtainedInitialImages = False
        self.obtainedIntrinsics = False

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

        rospy.loginfo(f"[{rospy.get_name()}] " + "Loaded params. Waiting for images and intrinsics to start publishing")
        self.start()
        rospy.loginfo(f"[{rospy.get_name()}] " + "Server is up")



    def getImages(self, color, depth):
        self.depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        self.color_img = self.bridge.imgmsg_to_cv2(color, desired_encoding="bgr8")
        self.obtainedInitialImages = True
        

    def getCamInfo(self,msg):
        self.intrinsics = np.array(msg.K).reshape(3,3)
        self.obtainedIntrinsics = True
        rospy.logdebug(f"[{rospy.get_name()}] " + "Obtained camera intrinsics")
        self.cameraInfoSub.unregister()

    def getBboxes(self,msg):
        
        num_detections = (msg.nPredictions)
        self.bboxes = {
            "boxes" : np.array(msg.box_bounding_boxes).reshape(num_detections,4),
            "box_classes" : np.array(msg.box_classes).reshape(num_detections)
        }
        # if self.depth_img is None:
        #     return
        # msg = self.bboxes
        # loc = np.where(np.array(msg['box_classes']).astype(np.uint8) == 60)[0]
        # if len(loc) > 0:
      
        #     upscale_fac = self.image_scale
        #     loc = loc[0]
        #     box = np.squeeze(msg['boxes'][loc])
        #     y1, x1, y2, x2 =  box
        #     crop = self.depth_img[int(y1 * upscale_fac) : int(y2 * upscale_fac), int(x1 * upscale_fac) : int(x2 * upscale_fac)]
        #     color_crop = self.color_img[int(y1 * upscale_fac) : int(y2 * upscale_fac), int(x1 * upscale_fac) : int(x2 * upscale_fac)]
        #     h, w = crop.shape[:2]
        #     xmap = np.arange(0, w) + x1 * upscale_fac
        #     ymap = np.arange(0, h) + y1 * upscale_fac
        #     xmap, ymap = np.meshgrid(xmap, ymap)
        #     points_z = crop / 1000.0
        #     cx, cy, fx, fy = self.intrinsics[0][2], self.intrinsics[1][2], self.intrinsics[0][0], self.intrinsics[1][1]
        #     points_x = (xmap - cx) * points_z / fx
        #     points_y = (ymap - cy) * points_z / fy
        #     cloud = np.stack([points_x, points_y, points_z], axis=-1)
        #     pcd_points = cloud.reshape([-1, 3])
        #     pcd_points = np.hstack((pcd_points,np.ones((pcd_points.shape[0],1)))) 
        #     transform_mat = self.getTransform('/base_link', '/camera_depth_optical_frame')            
        #     pcd_points = np.matmul(transform_mat,pcd_points.T).T  

        #     #filter ground plane
        #     pcd_points = pcd_points[pcd_points[:,2]>0.2]

            #visualize point clouds using o3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pcd_points[:, :3])
            # o3d.visualization.draw_geometries([pcd])

            # print(np.median(pcd_points[:,0]), pcd_points.shape)

            # msg = PointCloud2()

            # msg.header.frame_id = "base_link"
            # msg.height = 1
            # msg.width = pcd_points.shape[0]
            # msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
            #             PointField('y', 4, PointField.FLOAT32, 1),
            #             PointField('z', 8, PointField.FLOAT32, 1)]
            # msg.is_bigendian = False
            # msg.point_step = 12
            # msg.row_step = msg.point_step * msg.width
            # msg.data = pcd_points[:, :3].tobytes()
            # self.temppub.publish(msg)



    def getTransform(self, target_frame, source_frame):

        # if self.tf_listener.frameExists(target_frame) and self.tf_listener.frameExists(source_frame):
        if True: #TODO(@praveenvnktsh) DOWNGRADE THE PACKAGE. THIS IS A HAC
            t = self.tf_listener.getLatestCommonTime(target_frame, source_frame)
            position, quaternion = self.tf_listener.lookupTransform(target_frame, source_frame, t)

            #if you face error that transform not found. launch stretch_driver before running realsense node.
            #realsense node publishes o tf_static and it overrides all tf variables for some reason.
            #the distance between link head pan to base link is 1.35 in gazebo but 1.32 through tf transforms.

            tf_matrix = self.tf_ros.fromTranslationRotation(position,quaternion)

            # print("base2cam=",tf_matrix)

            return tf_matrix
        rospy.logerr(f"[{rospy.get_name()}] " + "Transform not found between {} and {}".format(target_frame, source_frame))
        return None

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
        if(vis_flag):
            grasp_obj.vis_grasps(gg, cloud)
        else:
            grasp_obj.sort_grasps(gg)

        return gg

    def inference(self, goal):
        self.objectOfInterest = goal.request
        box_dict = self.bboxes
        rospy.loginfo(f"[{rospy.get_name()}] " + "grasp generation failed since transforms are unavailable.")

        source_frame = self.transform_source_frame
        target_frame = self.transform_target_frame

        graspPose = GraspPoseResult()
        tf_mat_base2cam = self.getTransform(target_frame,source_frame)
        if tf_mat_base2cam is None:
            rospy.logerr(f"[{rospy.get_name()}] " + "grasp generation failed since transforms are unavailable.")
            graspPose.header.stamp = rospy.Time.now()
            graspPose.success = False
            self._as.set_succeeded(graspPose)
            return

        mask = None
        for idx,label in enumerate(box_dict["box_classes"]):
            if(int(label)==self.objectOfInterest):
                mask = np.array(box_dict["boxes"][idx]).astype(np.int32)*self.image_scale
        
        if mask is None:
            rospy.logerr(f"[{rospy.get_name()}] " + "grasp generation failed since object is not in field of view of the robot.")
            graspPose.header.stamp = rospy.Time.now()
            graspPose.success = False
            self._as.set_succeeded(graspPose)
            return

        color_img = self.color_img
        depth_img = self.depth_img

        graspNetFailed = False
        if self.mode == 'graspnet':
            rospy.loginfo(f"[{rospy.get_name()}] " + "Running GraspNet")
            graspObj = self.runGraspnet(
                color = color_img, 
                depth = depth_img, 
                bbox = mask,
                intrinsics = self.intrinsics,
                vis_flag = False
            )
            if len(graspObj) == 0:
                graspNetFailed = True
                rospy.loginfo(f"[{rospy.get_name()}] " + "Graspnet failed to find a grasp. Falling back to centroid")
            else:
                rospy.loginfo(f"[{rospy.get_name()}] " + "Found grasp from network!")
                rot_mat_grasp = graspObj[0].rotation_matrix.astype(np.float32)
                t_grasp = graspObj[0].translation.astype(np.float32)

                tf_mat_cam2obj = np.hstack((rot_mat_grasp,t_grasp.reshape((3,1))))
                tf_mat_cam2obj = np.vstack((tf_mat_cam2obj,np.array([[0,0,0,1]])))

                tf_base2obj = np.matmul(tf_mat_base2cam,tf_mat_cam2obj)
                
                
                graspPose.x = tf_base2obj[0][3]
                graspPose.y = tf_base2obj[1][3]
                graspPose.z = tf_base2obj[2][3]
                #yaw set to 0 since robot can't grasp in any other orientation
                graspPose.yaw = 0
                # graspPose.yaw = tf.transformations.euler_from_matrix(tf_base2obj)[2]
                self.publish_transform(tf_base2obj)
            

        if graspNetFailed or self.mode == 'centroid':
            rospy.loginfo(f"[{rospy.get_name()}] " + "Running Centroid")
            graspObj_centroid = self.grasp_obj.centroid_grasp(color_img, depth_img, mask, self.intrinsics)
            graspObj_centroid = np.vstack((graspObj_centroid.reshape(3,1),np.array([[1]])))
            tf_base2obj = np.matmul(tf_mat_base2cam, graspObj_centroid)
            graspPose.x = tf_base2obj[0][0]
            graspPose.y = tf_base2obj[1][0]
            graspPose.z = tf_base2obj[2][0]
            graspPose.yaw = 0
            mat = np.eye(4)
            mat[:3, 3] = tf_base2obj[0:3].flatten()
            # TODO: send transform here
            print(mat)
            self.publish_transform(mat)

        rospy.loginfo(f"[{rospy.get_name()}] " + 
            "Grasp Pose: " + f"{[graspPose.x, graspPose.y, graspPose.z, graspPose.yaw]}"
        )
        graspPose.header.stamp = rospy.Time.now()
        graspPose.success = True
        
        self._as.set_succeeded(graspPose)

    def start(self):
        while not self.obtainedInitialImages and not self.obtainedIntrinsics:
            rospy.logdebug(f"[{rospy.get_name()}] " + " Waiting to start server")
            rospy.sleep(1)

        self._as.start()
        rospy.loginfo(f"[{rospy.get_name()}] " + " Grasp Pose Server Started")

        
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