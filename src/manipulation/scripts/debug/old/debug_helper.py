#!/usr/bin/env python3

# TODO:
# Add feedbacks (debug)
# placing plane detection.
# Is grasp complete?

import time
import numpy as np
from tf import TransformerROS
from tf import TransformListener, TransformBroadcaster
# ROS action finite state machine
from control_msgs.msg import FollowJointTrajectoryAction

import rospy
import actionlib
import json
# import action message types
from helpers import move_to_pose
import tf
if __name__=='__main__':
    rospy.init_node("CLOSE")
    trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    trajectoryClient.wait_for_server()
    # move_to_pose(trajectoryClient, {
    #     "lift;to": 0.4,
    # })
    # move_to_pose(trajectoryClient, {
    #     'arm;to' : 0.1,
    # })
    move_to_pose(trajectoryClient, {
        "lift;to": 0.6,
    }, asynchronous = False)
    move_to_pose(trajectoryClient, {
        'arm;to' : 0.5,
    }, asynchronous= False)
    move_to_pose(trajectoryClient, {
        'arm;to' : 0.0,
    }, asynchronous= True)

    # move_to_pose(trajectoryClient, {
    #     "stretch_gripper;to": -50,
    # })
    # move_to_pose(trajectoryClient, {
    #     "stretch_gripper;to": -50,
    # })



    # try:
    #     rospy.spin()
    # except rospy.ROSInterruptException:
    #     pass
      