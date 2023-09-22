#!/usr/bin/env python3
import rospy
import time
import actionlib
import os
import numpy as np
import csv  
from datetime import datetime
from control_msgs.msg import FollowJointTrajectoryAction
from std_srvs.srv import Trigger
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from helpers import move_to_pose
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import glob

states = []
def callback(msg):
    effort = msg.effort
    positions = msg.position
    velocities = msg.velocity
    name = msg.name
    states.append([name, effort, positions, velocities])

X = []  # input features
y = []  # target variable (outcome)

def go_to_grasp_pos_without_contact():

    move_to_pose(trajectory_client, {
        'lift;to': 0.84
    })
    rospy.sleep(3)

    move_to_pose(trajectory_client, {
            "arm;to": 0.3,
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

if __name__ == '__main__':

    import pickle
    with open('/home/hello-robot/ws/src/manipulation/scripts/grasp_success/model_new_data.pkl', 'rb') as f:
        d = pickle.load(f)
        model = d['model']
        scaler = d['scaler']
        feature_names = d['feature_names']
        joint_names = d['joint_names']
        interval = d['interval']

    rospy.init_node("asdf")
    sub = rospy.Subscriber('/alfred/joint_states', JointState, callback)
    startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    startManipService()
    trajectory_client = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    trajectory_client.wait_for_server()
    # print(states[-1][1][1])

    # go_to_grasp_pos_without_contact()
    # move_to_pose(trajectory_client, {
    #     'lift;to': 0.84
    # })

    move_to_pose(trajectory_client, {
        'stretch_gripper;to': -50
    })

    rospy.sleep(3)
    move_to_pose(trajectory_client, {
        'lift;to': 0.94
    })
    rospy.sleep(3)
    X = [[]]
    for k, dd in enumerate(states):
        if k % interval == 0:
            for i, name in enumerate(dd[0]):
                if name in joint_names:
                    feature_idx = feature_names[joint_names.index(name)]
                    X[-1].append(dd[feature_idx][i])
    # print(states[-1][1][1])
    X = np.array(X)
    X = scaler.transform(X)
    # print(model.predict(X))
    print(model.predict_proba(X))
    if sum(model.predict(X)) == 1:
        print("GRASPED OBJECT SUCCESSFULLY. I AM ", model.predict_proba(X)[0][1] * 100, "% CONFIDENT")
    else:
        print("GRASP FAILED. I AM", model.predict_proba(X)[0][0] * 100, "% CONFIDENT")
    
    move_to_pose(trajectory_client, {
        'lift;to': 0.84
    })
    rospy.sleep(3)
    move_to_pose(trajectory_client, {
        'stretch_gripper;to': 100
    })

    