#!/usr/bin/env python3
import rospy
import time
import actionlib
import os
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

efforts = []
states = []
isContact = False
def callback(msg):
    global efforts, isContact
    av_effort = 20
    av_effort_window_size = 3
    for i in range(len(msg.name)):
        if msg.name[i] == 'joint_lift':
            effort = msg.effort[i]
            states.append(msg.position[i])
            efforts.append(msg.effort[i])
            if len(efforts) < av_effort_window_size:
                continue

            av_effort = sum(efforts[-av_effort_window_size:]) / av_effort_window_size
            # print(av_effort, effort)
            if av_effort < 5:
                isContact = True
                print("Lift effort is too low: {}, there is contact".format(effort))
                # move_to_pose(trajectory_client, {
                #     'lift;to': 1.0,  
                # })
                # rospy.sleep(3)
                
def move_until_contact(trajectory_client, pose):
    global isContact, states
    curz = states[-1]
    while not isContact or curz >= pose['lift;to']:
        curz = states[-1]
        print(curz, pose['lift;to'], isContact)
        if isContact:
            break
        move_to_pose(trajectory_client, {
            'lift;to': curz - 0.01,
        })
        
        rospy.sleep(0.05)
    move_to_pose(trajectory_client, {
        "stretch_gripper;to": 100,
        })



if __name__ == '__main__':
    rospy.init_node("asdf")
    rospy.Subscriber('/alfred/joint_states', JointState, callback)
    startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    startManipService()

    trajectory_client = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    trajectory_client.wait_for_server()
    move_to_pose(trajectory_client, {
        'lift;to': 1
    })
    time.sleep(3)
    move_until_contact(trajectory_client, {
        'lift;to' : 0.5
    })
    time.sleep(5)
    move_to_pose(trajectory_client, {
        'lift;to': 1
    })

    print(efforts)
    plt.plot(efforts)
    plt.ylabel("Lift effort")
    plt.xlabel("Time")
    plt.savefig('joint_effort.png')
    time.sleep(10)
