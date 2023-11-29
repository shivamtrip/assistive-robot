#!/usr/bin/env python3

"""
This script has functions related to control the base to visually-align to the object
"""


import rospy
import time
import numpy as np
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal


class AlignToObject:
    def __init__(self, trajectoryClient, human_detector):

        self.rate = 10.0
        self.kp = {
            'velx' : 0.7
        }
        self.kd =   {
            'velx' : 0.1
        }         

        self.x, self.y, self.z = 0, 0, 0
        self.obtainedInitialImages = False
        self.obtainedIntrinsics = False
        self.objectLocationInCamera = [np.inf, np.inf, np.inf]
        self.objectLocationArrInCamera = []

        self.trajectoryClient = trajectoryClient

        self.requestClearObject = False
        self.objectLocArr = []
        self.isDepthMatters = False
        self.human_detector = human_detector
        
        self.horizontal_alignment_offset = 0.0     #0.055
        self.radius=0
        self.z=0
        self.prevxerr = 0    
        self.gotTransforms = False
        self.human_detector.isDetected = False 
        self.actionServer = None
        self.cameraParams = {}
        self.debug_out = True
        
        self.vs_range = [(-np.deg2rad(60), np.deg2rad(60)), np.deg2rad(5)] # [(left angle, right angle), stepsize]

        self.maxDistanceToMove = 0.0
        self.alignment_threshold = 15            # degrees


    def findAndOrientToObject(self):
        print("Scanning area")


        self.move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -10 * np.pi/180,
            'head_pan;to' : self.vs_range[0][0],
        })
        rospy.sleep(2)
        print("STARTING SEARCH")
        nRotates = 0

        prev_angle_to_go = None

        while True:
            for i, angle in enumerate(np.arange(self.vs_range[0][0], self.vs_range[0][1], self.vs_range[1])):
                self.move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : angle,
                })
                rospy.sleep(0.5) 

            
            objectLocs = np.array(self.human_detector.objectLocArr)

            if len(objectLocs) != 0:

                # compute some statistical measure of where the object is to make some more intelligent servoing guesses.
                angleToGo, x, y, z, radius,  = self.human_detector.computeObjectLocation(self.human_detector.objectLocArr)


                # If robot isn't rotating more than 3 degrees between iterations, break (to avoid being stuck in loop)
                if prev_angle_to_go and abs(angleToGo) - abs(prev_angle_to_go) <= 3:
                        print("Robot alignment is still the same since the previous iteration. Exiting loop.")
                        break


                # If robot is aligned less than 'alignment_threshold' degrees wrt to object - break, else keep rotating
                if (abs(angleToGo) + 0.28 < np.radians(self.alignment_threshold)):
                    print(f"Robot is aligned within {np.degrees(angleToGo)} degrees of the user")
                    break
                else:
                    print(f"Robot not yet aligned to the user. Currently at {np.degrees(angleToGo)} degrees with respect to the user")


                rospy.loginfo('Object found at angle: {}. Location = {}. Radius = {}'.format(angleToGo, (x, y, z), radius))

                # Rotate base to align with object
                if(angleToGo<0):
                    self.move_to_pose(self.trajectoryClient, {
                        'base_rotate;by' : angleToGo - 0.28,
                    })
                elif(angleToGo>0):
                    self.move_to_pose(self.trajectoryClient, {
                        'base_rotate;by' : angleToGo - 0.28,  #+0.13,
                    })
                else:
                    self.move_to_pose(self.trajectoryClient, {
                        'base_rotate;by' : angleToGo,
                    })       

                rospy.sleep(2)
                self.move_to_pose(self.trajectoryClient, {
                'head_pan;to' : 0,
                })
                rospy.sleep(2)


                # Reset the array of detections since the robot has now stopped rotating
                self.human_detector.requestClearObject = True
                objectLocs = np.array(self.human_detector.objectLocArr)
                rospy.sleep(5)

                
                # If robot is aligned less than 'alignment_threshold' degrees wrt to object - break, else keep rotating
                if len(objectLocs) != 0:
                    # compute some statistical measure of where the object is to make some more intelligent servoing guesses.
                    angleToGo, x, y, z, radius,  = self.human_detector.computeObjectLocation(self.human_detector.objectLocArr)

                    if (abs(angleToGo)< np.radians(self.alignment_threshold )):
                        print(f"Robot is aligned within {np.degrees(angleToGo)} degrees of the user")
                        break
                    else:
                        print(f"Robot not yet aligned to the user. Currently at {np.degrees(angleToGo)} degrees with respect to the user")

                self.human_detector.requestClearObject = True

                
            else:
                rospy.loginfo("Object not found, rotating base.")
                self.move_to_pose(self.trajectoryClient, {
                    'base_rotate;by' : -np.pi/2,
                    'head_pan;to' : -self.vs_range[0][0],
                })
                rospy.sleep(5)
                nRotates += 1

            if nRotates >= 3:
                return False


        self.move_to_pose(self.trajectoryClient, {
            'head_pan;to' : 0,
        })

        rospy.sleep(1)
        return True


    def moveTowardsObject(self):
        rospy.loginfo("Moving towards the object")
        lastDetectedTime = time.time()
        startTime = time.time()
        vel = 0.1
        intervalBeforeRestart = 10

        self.move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -10 * np.pi/180,
        })

 
        maxGraspableDistance = 1.4  #1.2

        distanceToMove = self.human_detector.computeObjectLocation(self.human_detector.objectLocArr)[4] - maxGraspableDistance
        rospy.loginfo("Object distance for YOLO = {}".format(self.human_detector.computeObjectLocation(self.human_detector.objectLocArr)[4]))
        
        if distanceToMove < 0:
            rospy.loginfo("Object is close enough. Not moving.")
            return True
# 

        while True:
            motionTime = time.time() - startTime
            distanceMoved = motionTime * vel

            self.human_detector.requestClearObject = True

            num_of_retries = 0

            objectLocs = np.array(self.human_detector.objectLocArr)

            if len(objectLocs) != 0:
                # compute some statistical measure of where the object is to make some more intelligent servoing guesses.
                angleToGo, x, y, z, radius,  = self.human_detector.computeObjectLocation(self.human_detector.objectLocArr)

                if (abs(angleToGo)> np.radians(self.alignment_threshold)):          # if object is beyond a certain angle from robot

                    print(f"Stopping forward motion since robot is not aligned to the user. Currently at {np.degrees(angleToGo)} degrees with respect to the user")

                    self.move_to_pose(self.trajectoryClient, {
                        'base_translate;vel' : 0.0,
                    })
                    
                    rospy.sleep(2)

                    if(angleToGo<0):
                        self.move_to_pose(self.trajectoryClient, {
                            'base_rotate;by' : angleToGo - 0.28,
                        })
                    elif(angleToGo>0):
                        self.move_to_pose(self.trajectoryClient, {
                            'base_rotate;by' : angleToGo - 0.14,  #+0.13,
                        })
                    else:
                        self.move_to_pose(self.trajectoryClient, {
                            'base_rotate;by' : angleToGo,
                        })  

                    rospy.sleep(2)

                    continue
                    
                
                print(f"Robot is aligned to user within {np.degrees(angleToGo)} degrees")

                self.move_to_pose(self.trajectoryClient, {
                    'base_translate;vel' : vel,
                })
   
                if self.human_detector.isDetected and self.human_detector.objectLocArr:
                    lastDetectedTime = time.time()
                    # print(f"Compared distance: {self.objectLocArr[-1][0]}\n")
                    if self.human_detector.objectLocArr[-1][0] < maxGraspableDistance:
                        rospy.loginfo("Object is close enough. Stopping.")
                        # rospy.loginfo("Object location = {}".format(self.human_detector.objectLocArr[-1]))
                        break
                if distanceMoved >= distanceToMove - 1.0:
                    if time.time() - lastDetectedTime > intervalBeforeRestart:
                        rospy.loginfo("Lost track of the object. Going back to search again.")
                        self.move_to_pose(self.trajectoryClient, {
                            'base_translate;vel' : 0.0,
                        })  
                        return False

                rospy.sleep(0.1)

            else:
                self.move_to_pose(self.trajectoryClient, {
                    'base_translate;vel' : 0.0,
                })  
                rospy.sleep(3)
                print("Increasing retry attempts")
                if num_of_retries > 6:
                    return False
                num_of_retries += 1
                continue


        self.move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : 0.0,
        })          
        return True


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


    def move_to_pose(self, trajectoryClient, pose, asynchronous=False, custom_contact_thresholds=False):
        joint_names = [key for key in pose]
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(0.0)

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = joint_names
        if not custom_contact_thresholds: 
            joint_positions = [pose[key] for key in joint_names]
            point.positions = joint_positions
            trajectory_goal.trajectory.points = [point]
        else:
            pose_correct = all([len(pose[key])==2 for key in joint_names])
            if not pose_correct:
                rospy.logerr("[ALFRED].move_to_pose: Not sending trajectory due to improper pose. custom_contact_thresholds requires 2 values (pose_target, contact_threshold_effort) for each joint name, but pose = {0}".format(pose))
                return
            joint_positions = [pose[key][0] for key in joint_names]
            joint_efforts = [pose[key][1] for key in joint_names]
            point.positions = joint_positions
            point.effort = joint_efforts
            trajectory_goal.trajectory.points = [point]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        trajectoryClient.send_goal(trajectory_goal)
        if not asynchronous: 
            trajectoryClient.wait_for_result()