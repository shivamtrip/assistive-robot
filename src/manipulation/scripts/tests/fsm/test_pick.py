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

from fsm import ManipulationManager


# class_list:
#   # - 'drawer'
#   - 'soda_can' # 0
#   - 'tissue_paper' #1
#   - 'toothbrush' #2
#   - 'tie' #3
#   - 'cell phone' #4
#   - 'banana' #5
#   - 'apple' #6
#   - 'orange' #7
#   - 'bottle' #8
#   - 'cup' #9
#   - 'teddy bear' #10
#   - 'remote' #11
#   - 'marker' #12
# #input  
# bool to_open 
# int64 aruco_id
# float32[] saved_position
# ---
# #output
# bool success
# float32[] saved_position
# ---
# #feedback
# int64 curState
# string curStateInfo

if __name__ == '__main__':
    from std_srvs.srv import Trigger, TriggerResponse
    
    # node = ManipulationManager()
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
    
    goal = FindAlignGoal()
    goal.objectId = 10
    
    align_client.send_goal(goal)
    align_client.wait_for_result()
    
    
    goal = PickTriggerGoal()
    goal.objectId = 10
    goal.use_planner = False

    pick_client.send_goal(goal)
    pick_client.wait_for_result()

    rospy.loginfo(f'{rospy.get_name()} : "Manipulation Finished"')


 