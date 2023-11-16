#!/usr/bin/env python3

import rospy
from helpers import move_to_pose
import time
import numpy as np
from enum import Enum
import cv2
from control_msgs.msg import FollowJointTrajectoryAction
from std_srvs.srv import Trigger
import actionlib
from aruco import ArUcoDetector
from manipulation.msg import TriggerAction, TriggerFeedback, TriggerResult, TriggerGoal

class State(Enum):
    SEARCH = 1
    
    MOVE_TO_OBJECT = 3
    ROTATE_90 = 4
    ALIGN_HORIZONTAL = 5
    COMPLETE = 6

    PICK = 8
    NAVIGATION = 9
    PLACE = 10


    SCANNING = 11      
    ALIGN_TO_OBJECT = 12          

    FAILED = -1

class Error(Enum):
    CLEAN = 0
    SCAN_FAILED = 1
    

class AlignToObject:
    def __init__(self, objectId):

        startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        startManipService()
        self.server = actionlib.SimpleActionServer('manipulation_fsm', TriggerAction, execute_cb=self.main, auto_start=False)


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
        
        self.vs_range = [(-np.deg2rad(60), np.deg2rad(60)), np.deg2rad(15)] # [(left angle, right angle), stepsize]

        self.x, self.y, self.z = 0, 0, 0
        self.obtainedInitialImages = False
        self.obtainedIntrinsics = False

        self.maxDistanceToMove = 0.0

        self.objectLocationInCamera = [np.inf, np.inf, np.inf]
        self.objectLocationArrInCamera = []

        self.trajectoryClient = actionlib.SimpleActionClient('/alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        print("Waiting for trajectory server")
        self.trajectoryClient.wait_for_server()


        self.movebot = ArUcoDetector()
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
            poses = []
            for i, angle in enumerate(np.arange(self.vs_range[0][0], self.vs_range[0][1], self.vs_range[1])):
                move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : angle,
                    # 'head_pan;to' : 0,
                })
                rospy.sleep(1) # wait for the head to move
                result = self.movebot.getArucoLocation()
                if result is not None:
                    poses.append(result.copy())

            # This part of the code stitches detections together.
            if len(poses) != 0:
                pose_est = np.mean(poses, axis = 0)
                x_est, y_est, z_est = pose_est
                angle =  np.arctan2(y_est, x_est)
                move_to_pose(self.trajectoryClient, {
                    'base_rotate;by' : angle,
                })
                move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : 0
                })
                rospy.loginfo(f"Rotated base by: {angle} degrees")
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
        lastFailedTime = 0
        startTime = time.time()
        movement_vel = 0.075
        intervalBeforeRestart = 10
        k=2

        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -30 * np.pi/180,
        })

        maxGraspableDistance = 0.77

        result = self.movebot.getArucoLocation()
        rospy.sleep(0.5)
        if result is None:
            print("Aruco not detected")
            return False
        
        distanceToMove = result[0] - maxGraspableDistance
        print("\ndistance to move: ",distanceToMove)

        if distanceToMove < 0:
            rospy.loginfo("\nObject is close enough already. The robot doesn't need to move further forward.")
            return True
        rospy.sleep(5)

        i = 1
        avg_vel = 0
        vel_sum = 0

        while True:
            
            # detect aruco
            result = self.movebot.getArucoLocation()

            # if detected, vel = x else 0
            if not result:
                print("NO RESULT!")
                vel = 0.0
            else:
                print("Found result")

                x, y, z = result
                if x < maxGraspableDistance:
                    rospy.loginfo("Object is close enough now. Stopping the robot.")
                    break
                vel = movement_vel


            # move forward with vel
            move_to_pose(self.trajectoryClient, {
                'base_translate;vel' : vel,
            })  



    def alignObjectForManipulation(self):
        move_to_pose(self.trajectoryClient, {
                'head_pan;to' : -np.pi/2,
            }
        )
        move_to_pose(self.trajectoryClient, {
                'base_rotate;by' : np.pi/2,
            }
        )
        rospy.sleep(2)
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

        xerr = np.inf
        rospy.loginfo("Aligning object horizontally")
        prevxerr = 0
        curvel = 0.0

        result = self.movebot.getArucoLocation()
        if result is not None:
            x, y, z = result
            xerr = (x - offset)
        else:
            return False
        prevsettime = time.time()
        while abs(xerr) > 0.008:

            result = self.movebot.getArucoLocation()
            if result is not None:
                x, y, z = result
                xerr = (x - offset)
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
            'base_translate;by' : -0.02,
        }) 
        move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : 0.0,
        }) 
   
        return True
    
    def recoverFromFailure(self):
        pass

    def main(self,  goal: TriggerGoal):
        
        self.state = State.SEARCH
        while True:
            if self.state == State.SEARCH:
                success = self.findAndOrientToObject()                
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                rospy.loginfo("\n Object found and oriented.\nAbout to check if object is too far.")
                rospy.sleep(2)
                self.state = State.MOVE_TO_OBJECT

            elif self.state == State.MOVE_TO_OBJECT:
                success = self.moveTowardsObject()
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                rospy.loginfo("\nAbout to align the robot.")
                rospy.sleep(2)
                self.state = State.ROTATE_90

            elif self.state == State.ROTATE_90:
                success = self.alignObjectForManipulation()
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()

                    return False
                rospy.loginfo("\nThe robot will now align itself to the object.")
                rospy.sleep(2)
                self.state = State.ALIGN_HORIZONTAL

            elif self.state == State.ALIGN_HORIZONTAL:
                self.movebot.manipulationMethods.move_to_pregrasp(self.trajectoryClient) # Received at height, opens the gripper and makes yaw 0
                ee_x, ee_y, ee_z = self.movebot.manipulationMethods.getEndEffectorPose() # Transforming from base coordinate to gripper coordinate and retrieving the displacement in x
                success = self.alignObjectHorizontal(offset = ee_x - 0.055) # Fixing the displacement in x was 0.07, now is 0.055
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                rospy.loginfo("\n Object found and aligned horizontally.")
                rospy.sleep(2)
                if goal.isPick:
                    self.state = State.PICK
                else:
                    self.state = State.PLACE

            elif self.state == State.PICK:
                success = self.movebot.fetch()
                if not success:
                    rospy.loginfo("Object not fetched.")
                    return False
                rospy.loginfo("Object fetched successfully.")

                self.state = State.COMPLETE
                break
                
            elif self.state == State.PLACE:
                success = self.movebot.place()
                if not success:
                    rospy.loginfo("Object not placed.")
                    return False
                rospy.loginfo("Object placed successfully.")
                self.state = State.COMPLETE
                break
                
            rospy.sleep(0.5)
      
        self.reset()
        return True
    

if __name__ == "__main__":
 
    rospy.init_node("visual_aruco", anonymous=True)
    node = AlignToObject(39)
    goal= TriggerGoal()
    # goal.isPick= True
    goal.isPick = False
    node.main(goal)
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass