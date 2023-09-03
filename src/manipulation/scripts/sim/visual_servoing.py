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
from helpers import armclient, headclient, diffdrive, gripperclient
import tf
import tf.transformations
from tf import TransformerROS
from tf import TransformListener, TransformBroadcaster
from cv_bridge import CvBridge, CvBridgeError
from helpers import convertPointToDepth
import cv2
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Vector3, Vector3Stamped, PoseStamped, Point, Pose, PointStamped
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
    def __init__(self, objectId):
        rospy.init_node("visual_servoing", anonymous= False)
        self.objectId = -1
        self.node_name = 'visual_alignment'
        self.state = State.SEARCH
        self.rate = 10.0
        self.kp = {
            'velx' : 0.5
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
        
        self.vs_range = [(-np.deg2rad(90), np.deg2rad(90)), np.deg2rad(5)] # [(left angle, right angle), stepsize]

        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.listener = tf.TransformListener()
        self.cameraInfoSub = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.getCamInfo)
        self.bridge = CvBridge()
        
        self.obtainedInitialImages = False
        self.obtainedIntrinsics = False

        self.maxDistanceToMove = 0.0

        self.requestClearObject = False
        self.objectLocs = []
        self.finalObjectLocation = [np.inf, np.inf, np.inf]
        self.downscale_ratio = 0.4

        # TODO(@praveenvnktsh) fix in rosparam
        self.depth_image_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image,self.depthCallback)
        self.boundingBoxSub = rospy.Subscriber("/object_bounding_boxes", Detections, self.parseScene)
        while not self.obtainedIntrinsics and not self.obtainedInitialImages:
            rospy.loginfo("Waiting for imagse")
            rospy.sleep(1)
        rospy.loginfo(f"[{self.node_name}]: Node initialized")

        armclient.wait_for_server()
        headclient.wait_for_server()
        gripperclient.wait_for_server()

    
    def setObjectId(self, objectId):
        self.objectId = objectId
    def debug(self, state, info, pa):
        pass
    
    def computeObjectLocation(self):
        objectLocs = np.array(self.objectLocs)
        angleToGo = np.mean(objectLocs[:, 0])
        weights = objectLocs[:, 4]
        x = np.average(objectLocs[:, 1], weights = weights)
        y = np.average(objectLocs[:, 2], weights = weights)
        z = np.average(objectLocs[:, 3], weights = weights)
        radius = np.sqrt(x**2 + y**2)
        return angleToGo, x, y, z, radius

    def findAndOrientToObject(self):
        print("Scanning area")
        self.objectLocs = []
        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -10 * np.pi/180,
            'head_pan;to' : self.vs_range[0][0],
        })
        rospy.sleep(2)
        print("STARTING SEARCH")
        while True:
            for i, angle in enumerate(np.arange(self.vs_range[0][0], self.vs_range[0][1], self.vs_range[1])):
                move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : angle,
                })
                rospy.sleep(0.5) # TODO(@praveenvnktsh): make sure that this cannot be made as a blocking call

            objectLocs = np.array(self.objectLocs)
            if len(objectLocs) != 0:
                # compute some statistical measure of where the object is to make some more intelligent servoing guesses.
                angleToGo, x, y, z, radius,  = self.computeObjectLocation()

                self.objectLocation = [x, y, z]
                self.requestClearObject = True

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
        print("Found object at ", angle)
        rospy.sleep(1)
        
        

    def moveTowardsObject(self):
        print("MOVING TOWARDS THE OBJECT")
        lastDetectedTime = time.time()
        startTime = time.time()
        vel = 0.1
        intervalBeforeRestart = 2.5
        distanceThreshold = 0.85
        self.requestClearObject = True
        while self.requestClearObject:
            rospy.sleep(0.1)

        while len(self.objectLocs) <= 10:
            rospy.sleep(0.1)

        maxGraspableDistance = 0.6

        if self.maxDistanceToMove < distanceThreshold:
            rospy.loginfo("Object is close enough. Not moving.")
            return True
        
        x, y, z = self.objectLocation
        distanceToMove = np.sqrt(x**2 + y**2)
        while True:
            motionTime = time.time() - startTime
            distanceMoved = motionTime * vel
            move_to_pose(self.trajectoryClient, {
                'base_translate;vel' : vel,
            })
            rospy.loginfo("Distance moved = {}".format(distanceMoved))


            if distanceMoved >= max(distanceToMove - 0.1, 0):
                rospy.loginfo("Almost reached the object. Starting closed loop control!")
                if self.isDetected:
                    lastDetectedTime = time.time()
                    if self.objectLocs[-1][0] < maxGraspableDistance:
                        rospy.loginfo("Object is close enough. Stopping.")
                        rospy.loginfo("Object location = {}".format(self.objectLocs[-1]))
                        break
                
                elif time.time() - lastDetectedTime > intervalBeforeRestart:
                    rospy.loginfo("Lost track of the object. Going back to search again.")
                    move_to_pose(self.trajectoryClient, {
                        'base_translate;vel' : 0.0,
                    })  
                    return False
                
                # elif (distanceMoved) > self.maxDistanceToMove  - distanceThreshold:
                #     rospy.loginfo("Reached the maximum distance to move. Going back to search again.")

                #     move_to_pose(self.trajectoryClient, {
                #         'base_translate;vel' : 0.0,
                #     })  
                #     return False
                # else:
                #     # use odometry to make some more intelligent guesses
                #     move_to_pose(self.trajectoryClient, {
                #         'base_translate;vel' : -0.1,
                #     })


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
        self.obtainedInitialImages=  True
        # to ensure thread safe behavior, writing happens only in this function on a separate thread
        if self.requestClearObject:
            self.requestClearObject = False
            self.objectLocs = []

        num_detections = (msg.nPredictions)
        msg = {
            "boxes" : np.array(msg.box_bounding_boxes).reshape(num_detections, 4),
            "box_classes" : np.array(msg.box_classes).reshape(num_detections),
            'confidences' : np.array(msg.confidences).reshape(num_detections),
        }
        loc = np.where(np.array(msg['box_classes']).astype(np.uint8) == self.objectId)[0]
        upscale_fac = 1/self.downscale_ratio
        if len(loc) > 0:
            loc = loc[0]
            box = np.squeeze(msg['boxes'][loc])
            confidence = np.squeeze(msg['confidences'][loc])



            x1, y1, x2, y2 =  box
            # print(x1, y1, x2, y2)
            crop = self.depth_image[int(y1 * upscale_fac) : int(y2 * upscale_fac), int(x1 * upscale_fac) : int(x2 * upscale_fac)]
            z = np.median(crop[crop != 0])/1000.0
            h, w = self.depth_image.shape[:2]
            
            y_new1 = w - x2 * upscale_fac
            x_new1= y1 * upscale_fac
            y_new2 = w - x1 * upscale_fac
            x_new2 = y2 * upscale_fac

            x_f = (x_new1 + x_new2)/2
            y_f = (y_new1 + y_new2)/2

            point = self.convertPointToBaseFrame(x_f, y_f, z)

            x, y, z = point.x, point.y, point.z

            radius = np.sqrt(x**2 + y**2)
            
            confidence /= radius
            # confidence *= ()
            if z > 0.7 : # to remove objects that are not in field of view
                self.isDetected = True
                detectedAngle = np.arctan2(y, x)
                self.objectLocs.append((detectedAngle, x, y, z, confidence))
            else:
                self.isDetected = False
                
        else:
            self.isDetected = False
        
    def getCamInfo(self,msg):
        self.intrinsics = np.array(msg.K).reshape(3,3)
        self.obtainedIntrinsics = True
        rospy.logdebug(f"[{rospy.get_name()}]" + "Obtained camera intrinsics")
        self.cameraInfoSub.unregister()

    
    def alignObjectHorizontal(self, offset = 0.0):
        self.requestClearObject = True
        while self.requestClearObject :
            rospy.sleep(0.1)
        while len(self.objectLocs) == 0:
            rospy.sleep(0.1)

        xerr = np.inf
        rospy.loginfo("Aligning object horizontally")
        prevxerr = np.inf
        while abs(xerr) > 0.005:
            angle, x, y, z, confidence = self.objectLocs[-1]
            xerr = (x) + offset
            print("error = ", xerr)

            dxerr = (xerr - prevxerr)
            vx = (self.kp['velx'] * xerr + self.kd['velx'] * dxerr)
            prevxerr = xerr
            move_to_pose(self.trajectoryClient, {
                    'base_translate;vel' : vx,
            }) 
            
            rospy.sleep(0.1)

        move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : 0.0,
        }) 

    def depthCallback(self, depth_img):
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        depth_img = np.array(depth_img, dtype=np.float32)
        self.depth_image = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
        

    def main(self, objectId):
        self.objectId = objectId
        print("Triggered Visual Servoing")
        while True:
            if self.state == State.SEARCH:
                self.findAndOrientToObject()                
                self.state = State.MOVE_TO_OBJECT
            elif self.state == State.MOVE_TO_OBJECT:
                if self.moveTowardsObject() == False:
                    self.findAndOrientToObject()
                    self.state = State.SEARCH
                else:
                    self.state = State.ROTATE_90
            elif self.state == State.ROTATE_90:
                self.alignObjectForManipulation()
                self.state = State.ALIGN_HORIZONTAL
            elif self.state == State.ALIGN_HORIZONTAL:
                self.alignObjectHorizontal()
                self.state = State.COMPLETE
            elif self.state == State.COMPLETE:
                break
            rospy.sleep(0.5)
        print("Visual Servoing Complete")
    

if __name__ == "__main__":
    # unit test
    # switch_to_manipulation = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    # switch_to_manipulation.wait_for_service()
    # switch_to_manipulation()
    node = AlignToObject(39)
    
    
    node.main(39)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass