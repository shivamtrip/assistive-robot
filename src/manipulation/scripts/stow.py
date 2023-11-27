import numpy as np
import rospy
import tf
# from scene_parser import SceneParser
from scene_parser_refactor import SceneParser
from visual_servoing import AlignToObject
from manip_basic_control import ManipulationMethods
import actionlib
from manipulation.msg import StowTriggerAction, StowTriggerFeedback, StowTriggerResult, StowTriggerGoal
from helpers import move_to_pose
import time
from enum import Enum
import actionlib
import open3d as o3d
from std_srvs.srv import Trigger, TriggerResponse

class StowManager():
    def __init__(self, scene_parser : SceneParser, trajectory_client, manipulation_methods : ManipulationMethods):
        self.node_name = 'stow_manager'
        self.scene_parser = scene_parser
        self.trajectory_client = trajectory_client
        self.manipulationMethods = manipulation_methods
        
        self.server = actionlib.SimpleActionServer('stow_action', StowTriggerAction, execute_cb=self.execute_cb, auto_start=False)
        
        rospy.loginfo("Waiting for stow robot service.")
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        self.stow_robot_service.wait_for_service()
        
        self.server.start()
        
        rospy.loginfo("Loaded stow manager.")    
    
    
    def stow(self, use_planner = False):
        success = True
        if not use_planner:
            rospy.loginfo("Stowing robot (no plan)")
            self.stow_robot_service()
        else:
            rospy.loginfo("Planning and stowing.")
            self.manipulationMethods.plan_and_stow()
            
        return success
    
    def execute_cb(self, goal : StowTriggerGoal):
        starttime = time.time()
        self.goal = goal
        
        success = self.stow(use_planner=goal.use_planner)
        
        rospy.loginfo("Stow action took " + str(time.time() - starttime) + " seconds.")
        result = StowTriggerResult()
        result.success = success
        if success:
            self.server.set_succeeded(result)
        else:
            self.server.set_aborted(result)