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
import json
from control_msgs.msg import FollowJointTrajectoryAction
from plane_detector.msg import PlaneDetectAction, PlaneDetectResult, PlaneDetectGoal
from helpers import move_to_pose
from manipulation.msg import PickTriggerAction, PickTriggerFeedback, PickTriggerResult, PickTriggerGoal
from fsm import ManipulationFSM


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


if __name__ == '__main__':
    from std_srvs.srv import Trigger, TriggerResponse
    startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    startManipService.wait_for_service()
    startManipService()
    class_list = rospy.get_param('/object_detection/class_list')
    
    stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
    stow_robot_service.wait_for_service()
    
    stow_robot_service()
    node = ManipulationFSM()
    client = actionlib.SimpleActionClient('pick_action', PickTriggerAction)
    print("Waiting to get pick_action server")
    client.wait_for_server()
    goal = PickTriggerGoal()
    goal.objectId = 6
    client.send_goal(goal)
    client.wait_for_result()
    rospy.loginfo(f'{rospy.get_name()} : "Manipulation Finished"')


 