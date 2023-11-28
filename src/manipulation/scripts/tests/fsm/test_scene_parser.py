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

from scene_parser_refactor import SceneParser

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
    rospy.init_node("temp")
    scene_parser = SceneParser()
    rospy.spin()


 