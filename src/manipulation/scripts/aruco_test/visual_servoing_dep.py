#!/usr/bin/env python3

import rospy
from helpers import move_to_pose
import time
import numpy as np
from enum import Enum
import cv2
from control_msgs.msg import FollowJointTrajectoryAction
import actionlib
from main import MoveRobot
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

        self.x, self.y, self.z = 0, 0, 0
        self.obtainedInitialImages = False
        self.obtainedIntrinsics = False

        self.maxDistanceToMove = 0.0

        self.objectLocationInCamera = [np.inf, np.inf, np.inf]
        self.objectLocationArrInCamera = []

        self.trajectoryClient = actionlib.SimpleActionClient('/alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        print("Waiting for trajectory server")
        self.trajectoryClient.wait_for_server()


        self.movebot = MoveRobot()
        self.requestClearObject = False
        self.objectLocArr = []
        self.isDepthMatters = False
 


    def findAndOrientToObject(self):
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
            _,[x, y, z]=self.movebot.getArucoLocation()
            if(self.movebot.getArucoLocation()):
                break

            # objectLocs = np.array(self.objectLocArr)
            if self.movebot.getArucoLocation():

                _,[x, y, z]=self.movebot.getArucoLocation()
                self.objectLocation = [x, y, z]
                self.requestClearObject = True
                # self.maxDistanceToMove = radius
                    
                move_to_pose(self.trajectoryClient, {
                    'base_rotate;by' : np.arctan2(y,x),
                })
                move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : 0,
                })
                rospy.loginfo(f"Rotated base by: {np.arctan2(y,x)} degrees")
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
        # while self.requestClearObject:
        #     rospy.sleep(0.1)
        rospy.loginfo("Object location array cleared. Waiting for new object location")
        startTime = time.time()
        while len(self.objectLocArr) <= numberOfCopies:
            rospy.loginfo(f"Accumulated copies = {len(self.objectLocArr)} ")
            if time.time() - startTime >= 10:
                rospy.loginfo("Object not found. Restarting search")
  
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



        maxGraspableDistance = 0.77


        distanceToMove = self.objectLocation[1] - maxGraspableDistance
        if distanceToMove < 0:
            rospy.loginfo("Object is close enough. Not moving.")
            return True
        rospy.sleep(5)
        while True:
            motionTime = time.time() - startTime
            distanceMoved = motionTime * vel
            
            move_to_pose(self.trajectoryClient, {
                'base_translate;vel' : vel,
            })

            # rospy.loginfo("Object location = {}".format(self.objectLocArr[-1]))
            if 1:
                if self.movebot.getArucoLocation():
                    lastDetectedTime = time.time()
                    if self.objectLocation[1] < maxGraspableDistance:
                        rospy.loginfo("Object is close enough. Stopping.")
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
        # if not self.clearAndWaitForNewObject():
        #     return False

        xerr = 0.09
        rospy.loginfo("Aligning object horizontally")
        prevxerr = 0
        curvel = 0.0
        prevsettime = time.time()
        while abs(xerr) > 0.008:
            # x = self.objectLocation[0]
            _,[x,y,z]=self.movebot.getArucoLocation()
            # if self.isDetected:
            xerr = (x) + offset
            # else:
            #     rospy.logwarn("Object not detected. Using open loop motions")
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
   
        return True
    
    def recoverFromFailure(self):
        pass

    def main(self):
        
        self.state = State.SEARCH
        while True:
            if self.state == State.SEARCH:
                success = self.findAndOrientToObject()                
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                rospy.loginfo("\n Object found and oriented.")
                self.state = State.MOVE_TO_OBJECT

            elif self.state == State.MOVE_TO_OBJECT:
                success = self.moveTowardsObject()
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                rospy.loginfo("\n Object found and move successful.")
                self.state = State.ROTATE_90

            elif self.state == State.ROTATE_90:
                success = self.alignObjectForManipulation()
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()

                    return False
                rospy.loginfo("\n Object found and rotated.")
                self.state = State.ALIGN_HORIZONTAL

            elif self.state == State.ALIGN_HORIZONTAL:
                success = self.alignObjectHorizontal()
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                rospy.loginfo("\n Object found and aligned horizontally.")

                self.state = State.COMPLETE

            elif self.state == State.COMPLETE:
                break
            rospy.sleep(0.5)
        print("Visual Servoing Complete")
        self.reset()
        return True
    

if __name__ == "__main__":
 
    rospy.init_node("visual_aruco", anonymous=True)
    node = AlignToObject(39)
    node.main()
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass