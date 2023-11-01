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
from yolo.msg import Detections
from termcolor import colored

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
        self.rate = 10.0
        self.kp = {
            'velx' : 0.7
        }
        self.kd =   {
            'velx' : 0.1
        }         

        self.prevxerr = 0    
        self.gotTransforms = False
        self.isDetected = False 
        self.actionServer = None
        self.cameraParams = {}
        self.debug_out = True
        
        self.vs_range = [(-np.deg2rad(60), np.deg2rad(60)), np.deg2rad(5)] # [(left angle, right angle), stepsize]

        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.listener = tf.TransformListener()
        self.tf_ros = TransformerROS()
        self.cameraInfoSub = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.getCamInfo)
        self.bridge = CvBridge()
        
        self.obtainedInitialImages = False
        self.obtainedIntrinsics = False

        self.maxDistanceToMove = 0.0

        self.objectLocationInCamera = [np.inf, np.inf, np.inf]
        self.objectLocationArrInCamera = []

        self.requestClearObject = False
        self.objectLocArr = []
        self.upscale_fac = 1/rospy.get_param('/image_shrink/downscale_ratio')
        self.isDepthMatters = False
        # TODO(@praveenvnktsh) fix in rosparam
        self.depth_image_subscriber = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image,self.depthCallback)
        self.boundingBoxSub = rospy.Subscriber("/object_bounding_boxes", Detections, self.parseScene)
        # while not self.obtainedIntrinsics and not self.obtainedInitialImages:
            # rospy.loginfo("Waiting for imagse")
            # rospy.sleep(1)
        rospy.loginfo(f"[{self.node_name}]: Node initialized")

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

    def findAndOrientToObject(self):
        print("Scanning area")
        # self.objectLocArr = []
        # if self.objectId == 60:
        #     self.isDepthMatters = True
        #     self.clearAndWaitForNewObject(2)
        #     self.isDepthMatters = False
        # else:
        # self.clearAndWaitForNewObject()

        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -10 * np.pi/180,
            'head_pan;to' : self.vs_range[0][0],
        })
        rospy.sleep(2)
        print("STARTING SEARCH")
        nRotates = 0
        while True:
            for i, angle in enumerate(np.arange(self.vs_range[0][0], self.vs_range[0][1], self.vs_range[1])):
                move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : angle,
                })
                rospy.sleep(0.5) # wait for the head to move

            objectLocs = np.array(self.objectLocArr)
            if len(objectLocs) != 0:
                # compute some statistical measure of where the object is to make some more intelligent servoing guesses.
                angleToGo, x, y, z, radius,  = self.computeObjectLocation(self.objectLocArr)

                self.objectLocation = [x, y, z]
                self.requestClearObject = True
                # self.maxDistanceToMove = radius

                rospy.loginfo('Object found at angle: {}. Location = {}. Radius = {}'.format(angleToGo, (x, y, z), radius))

                move_to_pose(self.trajectoryClient, {
                    'base_rotate;by' : angleToGo,
                })
                move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : 0,
                })
                
                break
            else:
                rospy.loginfo("Object not found, rotating base.")
                move_to_pose(self.trajectoryClient, {
                    'base_rotate;by' : -np.pi/2,
                    'head_pan;to' : -self.vs_range[0][0],
                })
                rospy.sleep(5)
                nRotates += 1

            if nRotates >= 3:
                return False
        print("Found object at ", angle)
        rospy.sleep(1)
        return True

    def recoverFromFailure(self):
        rospy.loginfo("Attempting to recover from failure")        
        move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : -0.1,
        })
        rospy.sleep(1)
        move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : 0.0,
        })
        
    def clearAndWaitForNewObject(self, numberOfCopies = 10):
        rospy.loginfo("Clearing object location array")
        while self.requestClearObject:
            rospy.sleep(0.1)
        rospy.loginfo("Object location array cleared. Waiting for new object location")
        startTime = time.time()
        while len(self.objectLocArr) <= numberOfCopies:
            rospy.loginfo(f"Accumulated copies = {len(self.objectLocArr)} ")
            if time.time() - startTime >= 10:
                rospy.loginfo("Object not found. Restarting search")
                # self.recoverFromFailure()
                # self.findAndOrientToObject()
                return False
            rospy.sleep(1)

        return True

    def moveTowardsObject(self):
        rospy.loginfo("Moving towards the object")
        lastDetectedTime = time.time()
        startTime = time.time()
        vel = 0.1
        intervalBeforeRestart = 10

        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -30 * np.pi/180,
        })

        if self.objectId == 60:
            self.isDepthMatters = True
            if not self.clearAndWaitForNewObject(2):
                return False

            self.isDepthMatters = False
        else:
            if not self.clearAndWaitForNewObject():
                return False

        maxGraspableDistance = 0.77
        if self.objectId == 41:
            maxGraspableDistance = 1

        distanceToMove = self.computeObjectLocation(self.objectLocArr)[3] - maxGraspableDistance
        rospy.loginfo("Object distance = {}".format(self.computeObjectLocation(self.objectLocArr)[4]))
        # rospy.loginfo("Distance to move = {}. Threshold = {}".format(distanceToMove, distanceThreshold))
        if distanceToMove < 0:
            rospy.loginfo("Object is close enough. Not moving.")
            return True
        
        while True:
            motionTime = time.time() - startTime
            distanceMoved = motionTime * vel
            
            move_to_pose(self.trajectoryClient, {
                'base_translate;vel' : vel,
            })
            # rospy.loginfo("Distance to move = {}, Distance moved = {}".format(distanceToMove, distanceMoved))
            # rospy.loginfo()
            rospy.loginfo("Object location = {}".format(self.objectLocArr[-1]))
            if self.objectId != 60:
                if self.isDetected:
                    lastDetectedTime = time.time()
                    if self.objectLocArr[-1][0] < maxGraspableDistance:
                        rospy.loginfo("Object is close enough. Stopping.")
                        rospy.loginfo("Object location = {}".format(self.objectLocArr[-1]))
                        break
                if distanceMoved >= distanceToMove - 0.3:
                    if time.time() - lastDetectedTime > intervalBeforeRestart:
                        rospy.loginfo("Lost track of the object. Going back to search again.")
                        move_to_pose(self.trajectoryClient, {
                            'base_translate;vel' : 0.0,
                        })  
                        return False
            else: # open loop movement for the table
                if distanceMoved >= distanceToMove:
                    break

            rospy.sleep(0.1)

        move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : 0.0,
        })          
        return True



    def alignObjectForManipulation(self):
        print("Aligning object for manipulation")
        move_to_pose(self.trajectoryClient, {
                'head_pan;to' : -np.pi/2,
            }
        )
        move_to_pose(self.trajectoryClient, {
                'base_rotate;by' : np.pi/2,
            }
        )
        rospy.sleep(4)
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
    
    
    def getTransform(self, target_frame, source_frame):

        # if self.tf_listener.frameExists(target_frame) and self.tf_listener.frameExists(source_frame):
        if True: #TODO(@praveenvnktsh) DOWNGRADE THE PACKAGE. THIS IS A HAC
            t = self.listener.getLatestCommonTime(target_frame, source_frame)
            position, quaternion = self.listener.lookupTransform(target_frame, source_frame, t)

            tf_matrix = self.tf_ros.fromTranslationRotation(position,quaternion)

            # print("base2cam=",tf_matrix)

            return tf_matrix
        rospy.logerr(f"[{rospy.get_name()}] " + "Transform not found between {} and {}".format(target_frame, source_frame))
        return None


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
    
    def reset(self):
        self.isDetected = False
        self.objectLocArr = []
        self.objectLocationArrInCamera = []
        self.state = State.SEARCH
        self.requestClearObject = False

    
    def alignObjectHorizontal(self, offset = 0.0):
        self.requestClearObject = True
        if not self.clearAndWaitForNewObject():
            return False

        xerr = np.inf
        rospy.loginfo("Aligning object horizontally")
        prevxerr = np.inf
        curvel = 0.0
        prevsettime = time.time()
        while abs(xerr) > 0.008:
            x, y, z, confidence = self.objectLocArr[-1]
            if self.isDetected:
                xerr = (x) + offset
            else:
                rospy.logwarn("Object not detected. Using open loop motions")
                xerr = (x - curvel * (time.time() - prevsettime)) + offset
            rospy.loginfo("Alignment Error = " + str(xerr))
            dxerr = (xerr - prevxerr)
            vx = (self.kp['velx'] * xerr + self.kd['velx'] * dxerr)
            prevxerr = xerr
            move_to_pose(self.trajectoryClient, {
                    'base_translate;vel' : vx,
            }) 
            curvel = vx
            prevsettime = time.time()
            
            rospy.sleep(0.1)
        move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : 0.0,
        }) 
        # risk
        # if self.isDetected is False:
        #     return False
        return True
    
    def recoverFromFailure(self):
        pass

    def main(self, objectId):
        self.objectId = objectId
        print("Triggered Visual Servoing for object of interest:" , objectId)
        self.state = State.SEARCH
        while True:
            if self.state == State.SEARCH:
                success = self.findAndOrientToObject()                
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                self.state = State.MOVE_TO_OBJECT

            elif self.state == State.MOVE_TO_OBJECT:
                success = self.moveTowardsObject()
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                self.state = State.ROTATE_90

            elif self.state == State.ROTATE_90:
                success = self.alignObjectForManipulation()
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                self.state = State.ALIGN_HORIZONTAL

            elif self.state == State.ALIGN_HORIZONTAL:
                success = self.alignObjectHorizontal()
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                self.state = State.COMPLETE

            elif self.state == State.COMPLETE:
                break
            rospy.sleep(0.5)
        print("Visual Servoing Complete")
        self.reset()
        return True
    

if __name__ == "__main__":
    # unit test
    switch_to_manipulation = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    switch_to_manipulation.wait_for_service()
    switch_to_manipulation()
    rospy.init_node("align_to_object", anonymous=True)
    node = AlignToObject(39)
    # node.main(39)
    node.moveTowardsObject()
    # node.trackObjectHeadCam()
    # for i in range(20):
    #     node.trackObjectHeadCam()
    #     node.objectLocationArrInCamera = []
    #     rospy.sleep(1)
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass