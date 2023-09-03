#!/usr/bin/env python3

from control_msgs.msg import FollowJointTrajectoryGoal
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from helpers import move_to_pose
from std_msgs.msg import String
import json
from scipy.spatial.transform import Rotation as R

import numpy as np
from sensor_msgs.msg import JointState, CameraInfo, Image
from enum import Enum
import tf
from tf import TransformerROS
from cv_bridge import CvBridge, CvBridgeError
from helpers import convertPointToDepth
import cv2
from geometry_msgs.msg import Vector3, PointStamped
import pandas as pd
# ROS action finite state machine
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryResult
import actionlib
from yolo.msg import Detections
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
    def __init__(self):
        rospy.init_node('visual_alignment')
        self.objectId = 39
        self.state = State.SEARCH
        self.rate = 10.0
        self.kp = {
            'velx' : 0.005
        }
        self.kd =   {
            'velx' : 0.0
        }         
        self.x = np.inf
        self.y = np.inf
        self.z = np.inf
        self.prevxerr = 0    
        self.gotTransforms = False
        self.isDetected = False 
        self.actionServer = None
        self.cameraParams = {}
        self.debug_out = True
        self.vs_range = [-np.pi/4, np.pi/4]
        self.switch_to_manipulation = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        self.switch_to_navigation = rospy.ServiceProxy('/switch_to_navigation_mode', Trigger)
        self.switch_to_position = rospy.ServiceProxy('/switch_to_position_mode', Trigger)
        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.obtainedInitialImages = False
        self.listener = tf.TransformListener()
        self.tf_ros = TransformerROS()

        self.cameraInfoSub = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.getCamInfo)
        self.obtainedIntrinsics = False

    def setObjectId(self, objectId):
        self.objectId = objectId
    def debug(self, state, info, pa):
        pass
    
    def switchModes(self, toMode = 'navigation'):
    
        try:
            if toMode == 'navigation':
                resp1 = self.switch_to_navigation()
            else:
                resp1 = self.switch_to_manipulation()
            if resp1.success:
                print("Switched to " + toMode + " mode")
            else:
                print("Switch to " + toMode + " mode failed")
                self.switchModes(toMode)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))

    

    def getCamInfo(self,msg):
        self.intrinsics = np.array(msg.K).reshape(3,3)
        self.obtainedIntrinsics = True
        rospy.loginfo(f"[{rospy.get_name()}] " + "Obtained camera intrinsics")
        self.cameraInfoSub.unregister()
        
    def getTransform(self, target_frame, source_frame):
        if True: #TODO(@praveenvnktsh) DOWNGRADE THE PACKAGE. THIS IS A HAC
            t = self.listener.getLatestCommonTime(target_frame, source_frame)
            position, quaternion = self.listener.lookupTransform(target_frame, source_frame, t)

            #if you face error that transform not found. launch stretch_driver before running realsense node.
            #realsense node publishes o tf_static and it overrides all tf variables for some reason.
            #the distance between link head pan to base link is 1.35 in gazebo but 1.32 through tf transforms.
            
            tf_matrix = self.tf_ros.fromTranslationRotation(position,quaternion)

            return tf_matrix
        rospy.logerr(f"[{rospy.get_name()}] " + "Transform not found between {} and {}".format(target_frame, source_frame))
        return None
    

    def parseDetections(self, msg):
        num_detections = (msg.nPredictions)
        msg = {
            "boxes" : np.array(msg.box_bounding_boxes).reshape(num_detections,4),
            "box_classes" : np.array(msg.box_classes).reshape(num_detections)
        }
        loc = np.where(np.array(msg['box_classes']).astype(np.uint8) == self.objectId)[0]
        print(loc)
        if len(loc) > 0:
            loc = loc[0]
            self.box = np.squeeze(msg['boxes'][loc])
            x1, y1, x2, y2 =  self.box
            print(x1, y1, x2, y2)
            intrinsic = self.intrinsics
            fx = intrinsic[0][0]
            fy = intrinsic[1][1] 
            cx = intrinsic[0][2] 
            cy = intrinsic[1][2]

            print("Oldcoords=",x1, y1, x2, y2)
            
            
            # transform = self.getTransform('/base_link', '/camera_depth_optical_frame', )
            # print(self.depth_image.shape)
            try:
                crop = self.depth_image[int(y1 * 3) : int(y2 * 3), int(x1 * 3) : int(x2 * 3)]
                z = np.median(crop[crop != 0])/1000.0
                self.obtainedInitialImages=  True
            except:
                pass
            h, w = self.depth_image.shape[:2]
            x = ((x1 + x2)/2) * 3
            y = ((y1 + y2)/2) * 3
            y_new1 = w - x2*3
            x_new1= y1*3
            y_new2 = w - x1*3
            x_new2 = y2*3

            print("New coordinates=",x_new1, y_new1, x_new2, y_new2)
            print("value of z=",z)

            x_f = (x_new1 + x_new2)/2
            y_f = (y_new1 + y_new2)/2

            print(x_f,y_f)

            x_gb = (x_f - cx) * z/ fx
            y_gb = (y_f - cy) * z/ fy

            #plot depth image with rectangle
            # cpy = self.depth_image.copy().astype(np.uint8)
            # # cpy = cv2.resize(cpy, (0, 0), fx = 0.333, fy = 0.3)
            # cpy = cv2.rectangle(cpy, (int(x1 * 3), int(y1 * 3)), (int(x2 * 3), int(y2 * 3)), (0, 0, 0), 2)
            # # cpy = cv2.rectangle(cpy, (int(x1 ), int(y1 )), (int(x2 ), int(y2 )), (0, 0, 0), 2)
            # cv2.imshow('a', cpy)
            # cv2.waitKey(1)

            # x, y = y, x
            # x = (x - cx) * z/ fx
            # y = (y - cy) * z/ fy

            # print(x, y, z)

            # self.x, self.y, self.z = transform @ (np.array([[x, y, z, 1]]).T)
            camera_point = PointStamped()
            camera_point.header.frame_id = '/camera_depth_optical_frame'
            camera_point.point.x = x_gb
            camera_point.point.y = y_gb
            camera_point.point.z = z
            # print(transform @ (np.array([[x, y, z, 1]]).T))
            print(self.listener.transformPoint('base_link', camera_point).point)
            self.isDetected = True
        else:
            self.isDetected = False
   

    def depthCallback(self, depth_img):
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        depth_img = np.array(depth_img, dtype=np.float32)
        self.depth_image = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)

    def init(self):
        print("Initializing visual servoing node")
        self.node_name = 'visual_servoing'
        # rospy.init_node(self.node_name, anonymous=False)
        self.bridge = CvBridge()
        # TODO(@praveenvnktsh) fix in rosparam
        self.depth_image_subscriber = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image,self.depthCallback)
        self.boundingBoxSub = rospy.Subscriber("/object_bounding_boxes", Detections, self.parseDetections)
        while not self.obtainedIntrinsics:
            rospy.sleep(1)
        rospy.loginfo(f"[{self.node_name}]: Node initialized")

    def main(self, objectId):
        rospy.spin()

    def test(self, objectId):
        self.objectId = objectId
        self.alignObjectHorizontal()

    

if __name__ == "__main__":
    # unit test
    node = AlignToObject()
    node.init()
    
    node.main(39)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass