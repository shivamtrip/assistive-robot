#! /usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from manip_basic_control import ManipulationMethods
from visual_servoing import AlignToObject
import rospy
import actionlib
from enum import Enum
import stretch_body.robot as rb
import numpy as np
from grasp_detector.msg import GraspPoseAction, GraspPoseResult, GraspPoseGoal
from manipulation.msg import PickTriggerGoal, PlaceTriggerGoal, PickTriggerAction, PlaceTriggerAction
from manipulation.msg import DrawerTriggerAction, DrawerTriggerFeedback, DrawerTriggerResult, DrawerTriggerGoal
import json
from control_msgs.msg import FollowJointTrajectoryAction
from plane_detector.msg import PlaneDetectAction, PlaneDetectResult, PlaneDetectGoal
from helpers import move_to_pose
from manipulation.msg import FindAlignAction, FindAlignResult, FindAlignGoal


if __name__ == '__main__':
    from std_srvs.srv import Trigger, TriggerResponse
    
    rospy.init_node("test")
    
    startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    startManipService.wait_for_service()
    startManipService()
    class_list = rospy.get_param('/object_detection/class_list')
    
    stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
    stow_robot_service.wait_for_service()
    
    stow_robot_service()
    
    pick_client = actionlib.SimpleActionClient('pick_action', PickTriggerAction)
    pick_client.wait_for_server()
    
    place_client = actionlib.SimpleActionClient('place_action', PlaceTriggerAction)
    place_client.wait_for_server()
    
    drawer_client = actionlib.SimpleActionClient('drawer_action', DrawerTriggerAction)
    drawer_client.wait_for_server()
    
    align_client = actionlib.SimpleActionClient('find_align_action', FindAlignAction)
    align_client.wait_for_server()
    
    # goal = FindAlignGoal()
    # goal.objectId = 13
    
    # align_client.send_goal(goal)
    # align_client.wait_for_result()
    
    
    goal = PlaceTriggerGoal(
        heightOfObject = 0.2,
        place_location = [0.4, 0.4, 0.4],
        fixed_place = False,
        use_place_location = False,
        is_rotate = False
    )
    
    place_client.send_goal(goal)
    place_client.wait_for_result()
    rospy.loginfo(f'{rospy.get_name()} : "Manipulation Finished"')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

 