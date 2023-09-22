#!/usr/bin/env python3

# TODO:
# Add feedbacks (debug)
# placing plane detection.
# Is grasp complete?

import time
import numpy as np

# ROS action finite state machine

import rospy
import actionlib
import json
# import action message types
from helpers import move_to_pose

class ManipulationMethods:

    @staticmethod
    def debug(state, info):
        infojson = json.dumps(info)
        rospy.loginfo(f'[ManipulationMethods]: State = {state} | Info = {infojson}')
    
    @staticmethod
    def moveToPreGraspPose(trajectoryClient, x, y, z, theta):
        pass

    @staticmethod
    def pick(trajectoryClient, x, y, z, theta):
        move_to_pose(trajectoryClient, {
            "lift;to": z,
        })
        # rospy.sleep(7)
        move_to_pose(trajectoryClient, {
            "stretch_gripper;to": 100,
        })
        rospy.sleep(1)
        move_to_pose(trajectoryClient, {
            "arm;to": y,
        })
        rospy.sleep(1)
        
        rospy.sleep(1)
        move_to_pose(trajectoryClient, {
            "stretch_gripper;to": 0,
        })

        rospy.sleep(1)
        move_to_pose(trajectoryClient, {
            "lift;to": z + 0.10,
        })
        rospy.sleep(1)
        move_to_pose(trajectoryClient, {
           "arm;to" : 0,
           "wrist_yaw;to" : 0,
        })
        rospy.sleep(1)
        move_to_pose(trajectoryClient, {
            "lift;to": 0.3,
        })
        
        

    @staticmethod
    def place(trajectoryClient, x, y, z, theta):
        
        move_to_pose(trajectoryClient, {
            "lift;to": z,
        })
        

        rospy.sleep(7)
        
        move_to_pose(trajectoryClient, {
            "wrist_yaw;to": theta*(np.pi/180),
        })

        rospy.sleep(3)
        
        move_to_pose(trajectoryClient, {
            "arm;to": y,
        })
        rospy.sleep(7)
        
        move_to_pose(trajectoryClient, {
            "lift;to": z,
        })
        rospy.sleep(5)

        move_to_pose(trajectoryClient, {
            "stretch_gripper;to": 50,
        })

        rospy.sleep(4)
        
        move_to_pose(trajectoryClient, {
            "arm;to": 0,
        })
        rospy.sleep(7)
        move_to_pose(trajectoryClient, {
            "lift;to": 0.3,
        })
        rospy.sleep(6)
        
    
if __name__=='__main__':
    from helpers import armclient, headclient, diffdrive, gripperclient


    x = 0 # If x!=0 , then robot needs to be translated to be in line with the object of interest
    y = 0.5 # Arm extension
    z = 0.6 # Arm going up # Absolute value of the robot state wrt base # Maxes out at 1.10
    theta = 0 # Yaw, 0 for pointing towards the object
    rospy.init_node("manip", anonymous=False)
    armclient.wait_for_server()
    headclient.wait_for_server()
    gripperclient.wait_for_server()
    ManipulationMethods.pick(None, 0, y, z, theta)


    # try:
    #     rospy.spin()
    # except rospy.ROSInterruptException:
    #     pass
      