#!/usr/bin/env python3
import rospy
import time
import actionlib
import os
import numpy as np
import csv  
from datetime import datetime
import matplotlib.pyplot as plt
from control_msgs.msg import FollowJointTrajectoryAction
from std_srvs.srv import Trigger
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from helpers import move_to_pose
import matplotlib as mpl
mpl.use('Agg')

states = []
def callback(msg):

    effort = msg.effort
    positions = msg.position
    velocities = msg.velocity
    name = msg.name
    states.append([name, effort, positions, velocities])

def go_to_grasp_pos_without_contact():

    move_to_pose(trajectory_client, {
        'lift;to': 0.84
    })
    rospy.sleep(3)

    move_to_pose(trajectory_client, {
            "arm;to": 0.3,
        })
    rospy.sleep(5)

    move_to_pose(trajectory_client, {
            "wrist_yaw;to": np.pi,
        })
    rospy.sleep(5)

    move_to_pose(trajectory_client, {
            "stretch_gripper;to": 100,
        })

def go_to_grasp_pos_with_contact():

    moveUntilContact = True

    move_to_pose(trajectory_client, {
        'lift;to': 0.76
    })
    rospy.sleep(3)

    move_to_pose(trajectory_client, {
            "wrist_yaw;to": np.pi,
        })
    rospy.sleep(3)

    move_to_pose(trajectory_client, {
            "stretch_gripper;to": 100,
        })

    rospy.sleep(3)

    move_to_pose(trajectory_client, {
            "arm;to": 0.3,
        })
    rospy.sleep(3)

    #move until contact hardcoded for now!

    move_to_pose(trajectory_client, {
        'lift;to': 0.74
    })
    rospy.sleep(3)


            
if __name__ == '__main__':
    
    rospy.init_node("asdf")
    sub = rospy.Subscriber('/alfred/joint_states', JointState, callback)
    startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    startManipService()
    trajectory_client = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    trajectory_client.wait_for_server()

    go_to_grasp_pos_without_contact()

    # go_to_grasp_pos_with_contact()
    
    # move_to_pose(trajectory_client, {
    #     'stretch_gripper;to': -50
    # })

    # rospy.sleep(3)
    # move_to_pose(trajectory_client, {
    #     'lift;to': 0.94
    # })
    # rospy.sleep(3)

    # import json
    # filename = 'remote_fail_2'
    # with open(f'/home/hello-robot/ws/src/manipulation/scripts/grasp_success/newest_data/{filename}.json', 'w') as f:
    #     f.write(json.dumps(states))
    
    # move_to_pose(trajectory_client, {
    #     'lift;to': 0.76
    # })
    # rospy.sleep(3)
    # move_to_pose(trajectory_client, {
    #     'stretch_gripper;to': 100
    # })
    # rospy.sleep(3)
    # move_to_pose(trajectory_client, {
    #     'lift;to': 0.74
    # })
    # rospy.sleep(3)
