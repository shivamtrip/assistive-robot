#!/usr/bin/env python3

"""
This script has functions related to control the base to visually-align to the object
"""


import rospy
import time
import numpy as np
from manipulation_methods import ManipulationMethods
from aruco_detection import ArUcoDetector
    

class AlignToObject:
    def __init__(self, trajectoryClient, aruco_detector):

        self.rate = 10.0
        self.kp = {
            'velx' : 0.7
        }
        self.kd =   {
            'velx' : 0.1
        }         
        self.prevxerr = 0    
        self.isDetected = False 
        
        self.vs_range = [(-np.deg2rad(60), np.deg2rad(60)), np.deg2rad(15)] # [(left angle, right angle), stepsize]

        self.x, self.y, self.z = 0, 0, 0
        self.obtainedInitialImages = False
        self.obtainedIntrinsics = False


        self.objectLocationInCamera = [np.inf, np.inf, np.inf]
        self.objectLocationArrInCamera = []

        self.trajectoryClient = trajectoryClient


        self.aruco_detector = aruco_detector
        self.manipulation_methods = ManipulationMethods()
        self.requestClearObject = False
        self.objectLocArr = []
        self.isDepthMatters = False
        
        self.horizontal_alignment_offset = 0.055
 


    def findAndOrientToObject(self):
        self.manipulation_methods.move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -10 * np.pi/180,
            'head_pan;to' : self.vs_range[0][0],
        })
        rospy.sleep(2)
        print("STARTING SEARCH")
        nRotates = 0
        while True:
            poses = []
            for i, angle in enumerate(np.arange(self.vs_range[0][0], self.vs_range[0][1], self.vs_range[1])):
                self.manipulation_methods.move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : angle,
                    # 'head_pan;to' : 0,
                })
                rospy.sleep(1) # wait for the head to move
                result = self.aruco_detector.getArucoLocation()
                if result is not None:
                    poses.append(result.copy())

            # This part of the code stitches detections together.
            if len(poses) != 0:
                pose_est = np.mean(poses, axis = 0)
                x_est, y_est, z_est = pose_est
                angle =  np.arctan2(y_est, x_est)
                self.manipulation_methods.move_to_pose(self.trajectoryClient, {
                    'base_rotate;by' : angle,
                })
                self.manipulation_methods.move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : 0
                })
                rospy.loginfo(f"Rotated base by: {angle} degrees")
                break
            else:
                rospy.loginfo("Object not found, rotating base.")
                self.manipulation_methods.move_to_pose(self.trajectoryClient, {
                    'base_rotate;by' : -np.pi/2,
                    'head_pan;to' : -self.vs_range[0][0],
                })
                rospy.sleep(5)
                nRotates += 1

            if nRotates >= 5:
                return False
        rospy.sleep(1)

        return True


    def moveTowardsObject(self):
        rospy.loginfo("Moving towards the object")
        movement_vel = 0.075

        self.manipulation_methods.move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -30 * np.pi/180,
        })

        maxGraspableDistance = 0.77

        result = None
        first_print = False
        detection_start_time = time.time()
        detection_duration = 0

        while not result:
            result = self.aruco_detector.getArucoLocation()
            if not result and not first_print:
                print("Aruco not yet detected.")
                first_print = True

            detection_duration = time.time() - detection_start_time
            # return False

            if detection_duration > 10:
                print("Did not detect an ArUco even after 10 seconds.")
                return False

        print("Aruco has been detected!")
        
        distanceToMove = result[0] - maxGraspableDistance
        print("\ndistance to move: ",distanceToMove)

        if distanceToMove < 0:
            rospy.loginfo("\nObject is close enough already. The robot doesn't need to move further forward.")
            return True
        rospy.sleep(5)


        detection_start_time = time.time()
        detection_duration = 0
        first_print = False

        while True:
            
            # detect aruco
            result = self.aruco_detector.getArucoLocation()

            if not result:
                vel = 0.0

                if not first_print:
                    print("DID NOT detect Aruco")
                    first_print = True

                detection_duration = time.time() - detection_start_time

                if detection_duration > 5:
                    print("Did not detect an ArUco even after 5 seconds.")
                    return False

            else:
                x, y, z = result
                if x < maxGraspableDistance:
                    rospy.loginfo("Object is close enough now. Stopping the robot.")
                    break
                vel = movement_vel
                detection_start_time = time.time()
                detection_duration = 0
                first_print = False


            self.manipulation_methods.move_to_pose(self.trajectoryClient, {
                'base_translate;vel' : vel,
            })  

        return True


    def alignObjectForManipulation(self):
        self.manipulation_methods.move_to_pose(self.trajectoryClient, {
                'head_pan;to' : -np.pi/2,
            }
        )
        self.manipulation_methods.move_to_pose(self.trajectoryClient, {
                'base_rotate;by' : np.pi/2,
            }
        )
        rospy.sleep(2)
        return True

    
    def alignObjectHorizontal(self, trajectoryClient):
        self.requestClearObject = True

        self.manipulation_methods.move_to_pregrasp(trajectoryClient) # Received at height, opens the gripper and makes yaw 0
        ee_x, ee_y, ee_z = self.manipulation_methods.getEndEffectorPose() # Transforming from base coordinate to gripper coordinate and retrieving the displacement in x

        offset = ee_x - self.horizontal_alignment_offset

        xerr = np.inf
        rospy.loginfo("Aligning object horizontally")
        prevxerr = 0
        curvel = 0.0

        result = self.aruco_detector.getArucoLocation()
        if result is not None:
            x, y, z = result
            xerr = (x - offset)
        else:
            return False
        prevsettime = time.time()
        while abs(xerr) > 0.008:

            result = self.aruco_detector.getArucoLocation()
            if result is not None:
                x, y, z = result
                xerr = (x - offset)
            else:
                rospy.logwarn("Object not detected. Using open loop motions")
                xerr = (x - curvel * (time.time() - prevsettime)) + offset
            
            # rospy.loginfo("Alignment Error = " + str(xerr))
            dxerr = (xerr - prevxerr)
            vx = (self.kp['velx'] * xerr + self.kd['velx'] * dxerr)
            prevxerr = xerr
            self.manipulation_methods.move_to_pose(self.trajectoryClient, {
                    'base_translate;vel' : vx,
            }) 
            curvel = vx
            prevsettime = time.time()
            
            rospy.sleep(0.1)

        self.manipulation_methods.move_to_pose(self.trajectoryClient, {
            'base_translate;by' : -0.02,
        }) 
        self.manipulation_methods.move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : 0.0,
        }) 
   
        self.manipulation_methods.move_to_pose(self.trajectoryClient,{'head_tilt;to': -20 * np.pi/180,})

        return True