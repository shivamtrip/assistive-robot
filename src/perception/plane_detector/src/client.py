#!/usr/bin/env python3

from pathlib import Path
import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pcl
import time
from fit_plane import fit_plane
from error_func import ransac_error
from plot_result import plot_plane
import rospy
import actionlib
from scipy.spatial.transform import Rotation as R
from plane_detector.msg import PlaneDetectAction, PlaneDetectGoal


"""
Example Action Client script for Plane Detector
"""

def action_client():

    goal = PlaneDetectGoal()
    goal.request = 1
    # Fill in the goal here
    print("SENDING GOAL")
    client.send_goal(goal)
    client.wait_for_result()

    print("GOT RESULT")

    ans = client.get_result()
    print(ans)

    time.sleep(1)


if __name__ == "__main__":

    rospy.init_node('plane_detector_client')

    client = actionlib.SimpleActionClient('plane_detector', PlaneDetectAction)
    client.wait_for_server()
    print("Action server is up")

    action_client()
