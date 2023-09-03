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
            
if __name__ == '__main__':
    rospy.init_node("asdf")
    sub = rospy.Subscriber('/alfred/joint_states', JointState, callback)
    startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    startManipService()
    trajectory_client = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    trajectory_client.wait_for_server()
    move_to_pose(trajectory_client, {
        'lift;to': 0.84
    })
    rospy.sleep(3)
    move_to_pose(trajectory_client, {
        'stretch_gripper;to': -50
    })

    rospy.sleep(3)
    move_to_pose(trajectory_client, {
        'lift;to': 0.94
    })
    rospy.sleep(3)
    # import json
    # filename = 'thin_wire_success'
    # with open(f'/home/hello-robot/ws/src/manipulation/scripts/grasp_success/newdata/{filename}.json', 'w') as f:
    #     f.write(json.dumps(states))
    
    move_to_pose(trajectory_client, {
        'lift;to': 0.84
    })
    rospy.sleep(3)
    move_to_pose(trajectory_client, {
        'stretch_gripper;to': 100
    })

    