#! /usr/bin/env python3

""""
This script contains methods for reading and processing ArUco detections
"""


import rospy
from sensor_msgs.msg import CameraInfo
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import time
import tf
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger
# from manipulation.srv import ArUcoStatus, ArUcoStatusResponse
from enum import Enum
from tf import TransformerROS
from yolo.msg import Detections
from geometry_msgs.msg import PointStamped


class HumanDetector:

    def __init__(self):

        self.listener = tf.TransformListener()
        self.tf_ros = TransformerROS()
        self.cameraInfoSub = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.getCamInfo)
        self.bridge = CvBridge()
        
        self.obtainedInitialImages = False
        self.obtainedIntrinsics = False

        self.objectId = None
        self.objectLocationInCamera = [np.inf, np.inf, np.inf]
        self.objectLocationArrInCamera = []
        self.downscale_ratio = 0.4

        self.requestClearObject = False
        self.objectLocArr = []
        self.upscale_fac = 1/self.downscale_ratio
        self.isDepthMatters = False
        self.depth_image_subscriber = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image,self.depthCallback)
        self.boundingBoxSub = rospy.Subscriber("/object_bounding_boxes", Detections, self.detector_status_control)
        # while not self.obtainedIntrinsics and not self.obtainedInitialImages:
        #     rospy.loginfo("Waiting for images")
        #     rospy.sleep(1)    
        # rospy.loginfo(f"[{self.node_name}]: Node initialized")

        

    def computeObjectLocation(self, objectLocs):
        objectLocs = np.array(objectLocs)
        if len(objectLocs) == 0:
            return np.inf, np.inf, np.inf, np.inf, np.inf
        print(colored("hello", 'blue'))

        weights = objectLocs[:, 3]
        x = np.average(objectLocs[:, 0], weights = weights)
        y = np.average(objectLocs[:, 1], weights = weights)
        z = np.average(objectLocs[:, 2], weights = weights)
        angleToGo = np.arctan2(y, x)
        radius = np.sqrt(x**2 + y**2)
        return angleToGo, x, y, z, radius


    def convertPointToBaseFrame(self, x_f, y_f, z):
        intrinsic = self.intrinsics
        fx = intrinsic[0][0]
        fy = intrinsic[1][1] 
        cx = intrinsic[0][2] 
        cy = intrinsic[1][2]

        x_gb = (x_f - cx) * z/ fx
        y_gb = (y_f - cy) * z/ fy

        camera_point = PointStamped()
        camera_point.header.frame_id = '/camera_depth_optical_frame'
        camera_point.point.x = x_gb
        camera_point.point.y = y_gb
        camera_point.point.z = z
        point = self.listener.transformPoint('base_link', camera_point).point
        return point


    def parseScene(self, msg):
        if self.objectId is None:
            return
        self.obtainedInitialImages = True
        # to ensure thread safe behavior, writing happens only in this function on a separate thread
        if self.requestClearObject:
            self.requestClearObject = False
            self.objectLocArr = []

        num_detections = (msg.nPredictions)
        msg = {
            "boxes" : np.array(msg.box_bounding_boxes).reshape(num_detections, 4),
            "box_classes" : np.array(msg.box_classes).reshape(num_detections),
            'confidences' : np.array(msg.confidences).reshape(num_detections),
        }

        loc = np.where(np.array(msg['box_classes']).astype(np.uint8) == self.objectId)[0]
        upscale_fac = self.upscale_fac
        if len(loc) > 0:
            loc = loc[0]
            box = np.squeeze(msg['boxes'][loc])
            confidence = np.squeeze(msg['confidences'][loc])
            x1, y1, x2, y2 =  box
            # print(x1, y1, x2, y2)
            crop = self.depth_image[int(y1 * upscale_fac) : int(y2 * upscale_fac), int(x1 * upscale_fac) : int(x2 * upscale_fac)]
            
            
            

            # draw rectangel

            # im = self.depth_image.copy().astype(np.uint8)
            # cv2.rectangle(im, (int(x1 * upscale_fac), int(y1 * upscale_fac)), (int(x2 * upscale_fac), int(y2 * upscale_fac)), (0, 0, 0), 2)
            # im = cv2.resize(im, (w//2, h//2), interpolation=cv2.INTER_AREA)
            # cv2.imshow("a", im)
            # cv2.waitKey(1)
            if self.objectId == 60 and self.isDepthMatters:
                # h, w = crop.shape[:2]
                # xmap = (np.arange(0, w) + x1) * upscale_fac
                # ymap = (np.arange(0, h) + y1) * upscale_fac
                # xmap, ymap = np.meshgrid(xmap, ymap)
                # points_z = crop / 1000.0
                # cx, cy, fx, fy = self.intrinsics[0][2], self.intrinsics[1][2], self.intrinsics[0][0], self.intrinsics[1][1]
                # points_x = (xmap - cx) * points_z / fx
                # points_y = (ymap - cy) * points_z / fy
                # cloud = np.stack([points_x, points_y, points_z], axis=-1)
                # pcd_points = cloud.reshape([-1, 3])
                # pcd_points = np.hstack((pcd_points,np.ones((pcd_points.shape[0],1)))) 
                # transform_mat = self.get_transform('/base_link', '/camera_depth_optical_frame')            
                # pcd_points = np.matmul(transform_mat,pcd_points.T).T  
                # print(np.median(pcd_points[:,0]))
                upscale_fac = self.upscale_fac
                h, w = crop.shape[:2]
                xmap = np.arange(0, w) + x1 * upscale_fac
                ymap = np.arange(0, h) + y1 * upscale_fac
                xmap, ymap = np.meshgrid(xmap, ymap)
                points_z = crop / 1000.0
                cx, cy, fx, fy = self.intrinsics[0][2], self.intrinsics[1][2], self.intrinsics[0][0], self.intrinsics[1][1]
                points_x = (xmap - cx) * points_z / fx
                points_y = (ymap - cy) * points_z / fy
                cloud = np.stack([points_x, points_y, points_z], axis=-1)
                pcd_points = cloud.reshape([-1, 3])
                pcd_points = np.hstack((pcd_points,np.ones((pcd_points.shape[0],1)))) 
                transform_mat = self.getTransform('/base_link', '/camera_depth_optical_frame')            
                pcd_points = np.matmul(transform_mat,pcd_points.T).T  

                #filter ground plane
                pcd_points = pcd_points[pcd_points[:,2]>0.2]

                #visualize point clouds using o3d
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pcd_points[:, :3])
                # o3d.visualization.draw_geometries([pcd])

                print(np.median(pcd_points[:,0]), pcd_points.shape)
                self.isDetected = True
                self.objectLocArr.append((np.median(pcd_points[:,0]), np.median(pcd_points[:,1]), np.median(pcd_points[:,2]), confidence))
            else:
                z_f = np.median(crop[crop != 0])/1000.0
                h, w = self.depth_image.shape[:2]
                y_new1 = w - x2 * upscale_fac
                x_new1= y1 * upscale_fac
                y_new2 = w - x1 * upscale_fac
                x_new2 = y2 * upscale_fac

                x_f = (x_new1 + x_new2)/2
                y_f = (y_new1 + y_new2)/2
                # print("cam frame", x_f, y_f, z_f)
                point = self.convertPointToBaseFrame(x_f, y_f, z_f)

                x, y, z = point.x, point.y, point.z
                rospy.logdebug("Object detected at" + str((x, y, z)))

                radius = np.sqrt(x**2 + y**2)
                
                confidence /= radius
                # confidence *= ()
                self.objectLocationArrInCamera.append((x_f, y_f, z_f, 1))
                if self.objectId == 60 or (z > 0.7  and radius < 2.5): # to remove objects that are not in field of view
                    # print(x, y, z)
                    self.isDetected = True
                    self.objectLocArr.append((x, y, z, confidence))
                else:
                    self.isDetected = False
                
        else:
            self.isDetected = False

    def getCamInfo(self,msg):
        self.intrinsics = np.array(msg.K).reshape(3,3)
        self.cameraParams = {
            'width' : msg.width,
            'height' : msg.height,
            'focal_length' : self.intrinsics[0][0],
        }
        self.obtainedIntrinsics = True
        rospy.logdebug(f"[{rospy.get_name()}]" + "Obtained camera intrinsics")
        self.cameraInfoSub.unregister()


    def depthCallback(self, depth_img):
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        depth_img = np.array(depth_img, dtype=np.float32)
        self.depth_image = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)


    def detector_status_control(self, msg):

        if self.objectId:
            self.parseScene(msg)