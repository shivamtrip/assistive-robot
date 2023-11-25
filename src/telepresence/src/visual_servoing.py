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
 
        self.prevxerr = 0    
        self.gotTransforms = False
        self.isDetected = False 
        self.actionServer = None
        self.cameraParams = {}
        self.debug_out = True
        
        self.vs_range = [(-np.deg2rad(60), np.deg2rad(60)), np.deg2rad(5)] # [(left angle, right angle), stepsize]

        self.maxDistanceToMove = 0.0




    def findAndOrientToObject(self):
        print("Scanning area")


        self.move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -10 * np.pi/180,
            'head_pan;to' : self.vs_range[0][0],
        })
        rospy.sleep(2)
        print("STARTING SEARCH")
        nRotates = 0
        while True:
            for i, angle in enumerate(np.arange(self.vs_range[0][0], self.vs_range[0][1], self.vs_range[1])):
                self.move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : angle,
                })
                rospy.sleep(0.5) 

            objectLocs = np.array(self.objectLocArr)
            print(objectLocs,"\n")
            if len(objectLocs) != 0:
                # compute some statistical measure of where the object is to make some more intelligent servoing guesses.
                angleToGo, x, y, z, radius,  = self.human_detector.computeObjectLocation(self.objectLocArr)

                self.objectLocation = [x, y, z]
                self.requestClearObject = True
                # self.maxDistanceToMove = radius

                rospy.loginfo('Object found at angle: {}. Location = {}. Radius = {}'.format(angleToGo, (x, y, z), radius))

                self.move_to_pose(self.trajectoryClient, {
                    'base_rotate;by' : angleToGo,
                })
                self.move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : 0,
                })
                
                break
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
        print("Found object at ", angle)
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
        # else:
        #     if not self.clearAndWaitForNewObject():
        #         return False

        maxGraspableDistance = 0.77
        if self.objectId == 41:
            maxGraspableDistance = 1

        distanceToMove = self.human_detector.computeObjectLocation(self.objectLocArr)[3] - maxGraspableDistance
        rospy.loginfo("Object distance = {}".format(self.human_detector.computeObjectLocation(self.objectLocArr)[4]))
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
            # rospy.loginfo("Object location = {}".format(self.objectLocArr[-1]))
            if self.objectId != 60:
                if self.isDetected:
                    lastDetectedTime = time.time()
                    if self.objectLocArr[-1][0] < maxGraspableDistance:
                        rospy.loginfo("Object is close enough. Stopping.")
                        rospy.loginfo("Object location = {}".format(self.objectLocArr[-1]))
                        break
                if distanceMoved >= distanceToMove - 0.6:
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