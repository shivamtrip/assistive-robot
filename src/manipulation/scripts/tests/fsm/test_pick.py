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
from manipulation.msg import TriggerAction, TriggerFeedback, TriggerResult, TriggerGoal
import json
from control_msgs.msg import FollowJointTrajectoryAction
from plane_detector.msg import PlaneDetectAction, PlaneDetectResult, PlaneDetectGoal
from helpers import move_to_pose

from fsm import ManipulationFSM

if __name__ == '__main__':
    from std_srvs.srv import Trigger, TriggerResponse
    startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    startManipService.wait_for_service()
    startManipService()
    class_list = rospy.get_param('/object_detection/class_list')
    node = ManipulationFSM()
    goal = TriggerGoal()
    goal.objectId = class_list.index('remote')
    goal.isPick = True
    node.main(goal)
    rospy.sleep(5)

    rospy.loginfo(f'{rospy.get_name()} : "Manipulation Finished"')


 