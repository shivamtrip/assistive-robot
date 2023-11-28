#!/usr/local/lib/robot_env/bin/python3

from landmark_index import *
from lseg_map import *
from vis_tools import *

import rospy
import actionlib
import argparse
from collections import OrderedDict 
import os
import rospy
import cv2
import json
from utils.clip_mapping_utils import *
from scipy.spatial.transform import Rotation as R
from vlmaps_ros.msg import VLMapsAction,VLMapsGoal, VLMapsResult, VLMapsFeedback
import numpy as np

class vlmaps_primitive_caller():

    def __init__(self, labels: list) -> None:
        self.vlmaps_client = actionlib.SimpleActionClient('vlmaps_primitive_server', VLMaps_primitiveAction)

        # wait for server
        rospy.loginfo("Waiting for VLMAPS primitive server...")
        self.vlmaps_client.wait_for_server()
        rospy.loginfo("Connected to VLMAPS primitive server")

        # Initialize results
        self.results = None

        # set up the goal labels
        self.labels = labels

    def feedback_cb(self, feedback):
        pass

    def send_goal(self,labels):
        """ Send the goal to the action server"""
        goal = VLMapsGoal()
        labels = self.labels
        
        for label in labels:
            goal.labels.append(label)
        self.vlmaps_client.send_goal(goal, feedback_cb=self.feedback_cb)
        self.vlmaps_client.wait_for_result()
        result = self.vlmaps_client.get_result()

        rospy.loginfo("Result: {}".format(result))

        if(self.vlmaps_client.get_state() == actionlib.GoalStatus.SUCCEEDED):
            rospy.loginfo("Goal succeeded!!")

if __name__ == "__main__":
    rospy.init_node('vlmap_primitive_caller', anonymous=True)

    # get the labels
    LABELS = ["sofa", "potted plant", "sink", "refrigerator", "table"]
    vlmaps_caller = vlmaps_primitive_caller(LABELS)
    vlmaps_caller.send_goal(LABELS)