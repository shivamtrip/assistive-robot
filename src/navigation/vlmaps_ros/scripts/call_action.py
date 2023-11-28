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

class vlmaps_fsm():

    def __init__(self, labels: list) -> None:
        self.vlmaps_client = actionlib.SimpleActionClient('lseg_server', VLMapsAction)

        # wait for server
        rospy.loginfo("Waiting for VLMAPS server...")
        self.vlmaps_client.wait_for_server()
        rospy.loginfo("Connected to VLMAPS server")

        # Initialize results
        self.results = None

        # set up the goal labels
        self.labels = labels

    def feedback_cb(self, feedback):
        pass

    def send_goal(self,labels):
        """ Send the goal to the action server"""
        goal = VLMapsGoal()
        mask_list = []
        heatmaps_list = []
        labels = self.labels
        
        for label in labels:
            goal.labels.append(label)
        self.vlmaps_client.send_goal(goal, feedback_cb=self.feedback_cb)
        self.vlmaps_client.wait_for_result()
        result = self.vlmaps_client.get_result()

        for i in range(len(result.masks)):
            height_mask = result.masks[i].height
            width_mask = result.masks[i].width
            height_heatmap = result.heatmaps[i].height
            width_heatmap = result.heatmaps[i].width
            mask = np.frombuffer(result.masks[i].data, 
                                 dtype=np.uint8).reshape(height_mask, width_mask)
            heatmap = np.frombuffer(result.heatmaps[i].data, 
                                    dtype=np.float32).reshape(height_heatmap, width_heatmap)
            mask_list.append(mask)
            heatmaps_list.append(heatmap)

        if(self.vlmaps_client.get_state() == actionlib.GoalStatus.SUCCEEDED):
            rospy.loginfo("Goal succeeded")
            rospy.loginfo("Result: {}".format(len(mask_list)))

        mask_list = np.array(mask_list)
        heatmaps_list = np.array(heatmaps_list)

        self.results = {"masks": mask_list, "heatmaps": heatmaps_list}

if __name__ == "__main__":
    rospy.init_node('vlmaps_caller', anonymous=True)

    # get the labels
    LABELS = "table, chair, floor, sofa, bed, other"
    LABELS = LABELS.split(",")
    vlmaps_caller = vlmaps_fsm(LABELS)
    vlmaps_caller.send_goal(LABELS)