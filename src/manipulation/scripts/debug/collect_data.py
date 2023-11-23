#!/usr/bin/env python3

from control_msgs.msg import FollowJointTrajectoryGoal
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from helpers import move_to_pose
from std_msgs.msg import String, Header
import time
import json
import numpy as np
from sensor_msgs.msg import JointState, CameraInfo, Image
from enum import Enum
# from helpers import armclient, headclient, diffdrive, gripperclient
import tf
import tf.transformations
from tf import TransformerROS
from tf import TransformListener, TransformBroadcaster
from cv_bridge import CvBridge, CvBridgeError
from helpers import convertPointToDepth
import cv2
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Vector3, Vector3Stamped, PoseStamped, Point, Pose, PointStamped
from sensor_msgs.msg import PointCloud2, PointField
import pandas as pd
# ROS action finite state machine
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryResult
import actionlib
import message_filters
import os
class State(Enum):
    SEARCH = 1
    
    MOVE_TO_OBJECT = 3
    ROTATE_90 = 4
    ALIGN_HORIZONTAL = 5
    COMPLETE = 6

    # substates
    SCANNING = 11      
    ALIGN_TO_OBJECT = 12          

    FAILED = -1

class Error(Enum):
    CLEAN = 0
    SCAN_FAILED = 1
    

class AlignToObject:
    def __init__(self, objectId):
        self.objectId = objectId
        self.node_name = 'visual_alignment'
        self.state = State.SEARCH
        self.kp = {
            'velx' : 0.7
        }
        self.kd =   {
            'velx' : 0.1
        }         

        self.prevxerr = 0    
        self.isDetected = False 
        
        self.vs_range = [(-np.deg2rad(60), np.deg2rad(60)), np.deg2rad(5)] # [(left angle, right angle), stepsize]

        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.listener = tf.TransformListener()
        self.tf_ros = TransformerROS()


        rospy.loginfo(f"[{rospy.get_name()}]: Waiting for camera intrinsics...")
        msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=10)
        self.intrinsics = np.array(msg.K).reshape(3,3)
        self.cameraParams = {
            'width' : msg.width,
            'height' : msg.height,
            'intrinsics' : np.array(msg.K).reshape(3,3),
            'focal_length' : self.intrinsics[0][0],
        }
        rospy.loginfo(f"[{rospy.get_name()}]: Obtained camera intrinsics.")

        
        self.bridge = CvBridge()
        
        self.obtainedInitialImages = False

        self.maxDistanceToMove = 0.0

        self.objectLocationInCamera = [np.inf, np.inf, np.inf]

        self.requestClearObject = False
        self.objectLocArr = []
        self.upscale_fac = 1/rospy.get_param('/image_shrink/downscale_ratio')
        
        
        self.depth_image_subscriber = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image,)
        self.rgb_image_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image,)
        sync = message_filters.ApproximateTimeSynchronizer([self.depth_image_subscriber, self.rgb_image_subscriber], 10, 0.1, allow_headerless=True)
        sync.registerCallback(self.get_images)        
        
        while not self.obtainedInitialImages:
            rospy.sleep(0.1)
        rospy.loginfo(f"[{self.node_name}]: Initial images obtained.")
        rospy.loginfo(f"[{self.node_name}]: Node initialized")
        self.images = []
        os.makedirs('/home/hello-robot/alfred-autonomy/src/manipulation/scripts/debug/images/depth', exist_ok=True)
        os.makedirs('/home/hello-robot/alfred-autonomy/src/manipulation/scripts/debug/images/rgb', exist_ok=True)
        os.makedirs('/home/hello-robot/alfred-autonomy/src/manipulation/scripts/debug/images/pose', exist_ok=True)

    def computeObjectLocation(self, objectLocs):
        objectLocs = np.array(objectLocs)
        if len(objectLocs) == 0:
            return np.inf, np.inf, np.inf, np.inf, np.inf

        weights = objectLocs[:, 3]
        x = np.average(objectLocs[:, 0], weights = weights)
        y = np.average(objectLocs[:, 1], weights = weights)
        z = np.average(objectLocs[:, 2], weights = weights)
        angleToGo = np.arctan2(y, x)
        radius = np.sqrt(x**2 + y**2)
        return angleToGo, x, y, z, radius

    def findAndOrientToObject(self):
        
        rospy.loginfo(f"[{self.node_name}]: Finding and orienting to object")
        
        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -30 * np.pi/180,
            
        })
        move_to_pose(self.trajectoryClient, {
        'head_pan;to' : self.vs_range[0][0],
            
        })
        
        rospy.sleep(2)
        nRotates = 0
        for i, angle in enumerate(np.arange(self.vs_range[0][0], self.vs_range[0][1], self.vs_range[1])):
            move_to_pose(self.trajectoryClient, {
                'head_pan;to' : angle,
            })
            rospy.sleep(1.5) # wait for the head to move
            cv2.imwrite(f'/home/hello-robot/alfred-autonomy/src/manipulation/scripts/debug/images/rgb/{str(i).zfill(6)}.png', self.color_image)
            cv2.imwrite(f'/home/hello-robot/alfred-autonomy/src/manipulation/scripts/debug/images/depth/{str(i).zfill(6)}.tif', self.depth_image)
            with open(f'/home/hello-robot/alfred-autonomy/src/manipulation/scripts/debug/images/pose/{str(i).zfill(6)}.json', 'w') as f:
                json.dump({
                    'head_pan' : angle,
                    'head_tilt' : -30 * np.pi/180,
                    'transform' : self.get_transform('camera_depth_optical_frame', 'base_link' ).tolist(),
                    'intrinsics' : self.intrinsics.tolist()
                }, f)
            rospy.sleep(0.5)
                
        return True

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

    
    def get_transform(self, base_frame, camera_frame):

        while not rospy.is_shutdown():
            try:
                t = self.listener.getLatestCommonTime(base_frame, camera_frame)
                (trans,rot) = self.listener.lookupTransform(base_frame, camera_frame, t)

                trans_mat = np.array(trans).reshape(3, 1)
                rot_mat = R.from_quat(rot).as_matrix()
                transform_mat = np.hstack((rot_mat, trans_mat))

                return transform_mat
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

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
            crop = self.depth_image[int(y1 * upscale_fac) : int(y2 * upscale_fac), int(x1 * upscale_fac) : int(x2 * upscale_fac)]
            
            z_f = np.median(crop[crop != 0])/1000.0
            h, w = self.depth_image.shape[:2]
            y_new1 = w - x2 * upscale_fac
            x_new1= y1 * upscale_fac
            y_new2 = w - x1 * upscale_fac
            x_new2 = y2 * upscale_fac
            x_f = (x_new1 + x_new2)/2
            y_f = (y_new1 + y_new2)/2
            point = self.convertPointToBaseFrame(x_f, y_f, z_f)

            x, y, z = point.x, point.y, point.z
            rospy.logdebug("Object detected at" + str((x, y, z)))

            radius = np.sqrt(x**2 + y**2)
            
            confidence /= radius
            
            if (z > 0.7  and radius < 2.5): # to remove objects that are not in field of view
                self.isDetected = True
                self.objectLocArr.append((x, y, z, confidence))
            else:
                self.isDetected = False
        else:
            self.isDetected = False

    def get_images(self, depth_img, color_img):
        self.obtainedInitialImages = True
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        self.depth_image = np.array(depth_img, dtype=np.uint16)
        self.color_image = self.bridge.imgmsg_to_cv2(color_img, desired_encoding="passthrough")
    
    def reset(self):
        self.isDetected = False
        self.objectLocArr = []
        self.state = State.SEARCH
        self.requestClearObject = False
    def main(self, objectId):
        self.objectId = objectId
        print("Triggered Visual Servoing for object of interest:" , objectId)
        self.findAndOrientToObject()     
        return True
    

if __name__ == "__main__":
    # unit test
    switch_to_manipulation = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    switch_to_manipulation.wait_for_service()
    switch_to_manipulation()
    rospy.init_node("align_to_object", anonymous=True)
    node = AlignToObject(39)
    node.main(39)
