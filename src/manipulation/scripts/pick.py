import numpy as np
import rospy
import tf
# from scene_parser import SceneParser
from scene_parser_refactor import SceneParser
from visual_servoing import AlignToObject
from manip_basic_control import ManipulationMethods
import actionlib
from manipulation.msg import PickTriggerAction, PickTriggerFeedback, PickTriggerResult, PickTriggerGoal
from helpers import move_to_pose
import time
from enum import Enum
import actionlib

class States(Enum):
    IDLE = 0
    ALIGNING = 1
    MOVING_TO_OBJECT = 2
    REALIGNMENT = 3
    OPEN_DRAWER = 4
    CLOSE_DRAWER = 5
    COMPLETE = 6

class PickManager():
    def __init__(self, scene_parser : SceneParser, trajectory_client, manipulation_methods : ManipulationMethods):
        self.node_name = 'pick_manager'
        self.scene_parser = scene_parser
        self.trajectory_client = trajectory_client
        self.manipulationMethods = manipulation_methods
        
        self.server = actionlib.SimpleActionServer('pick_action', PickTriggerAction, execute_cb=self.execute_cb, auto_start=False)
        
        self.vs_range = rospy.get_param('/manipulation/vs_range') 
        self.vs_range[0] *= np.pi/180
        self.vs_range[1] *= np.pi/180
        
        
        self.offset_dict = rospy.get_param('/manipulation/offsets')
        self.isContactDict = rospy.get_param('/manipulation/contact')
        
        self.class_list = rospy.get_param('/object_detection/class_list')
        self.label2name = {i : self.class_list[i] for i in range(len(self.class_list))}
        self.server.start()
        
        rospy.loginfo("Loaded pick manager.")    
    
    

    def pick(self, objectId, use_planner = False, isPublish = True):
        
        rospy.loginfo("Picking object" + str(objectId))
        starttime = time.time()
        self.manipulationMethods.move_to_pregrasp(self.trajectory_client)
        self.scene_parser.set_point_cloud(publish = False, use_detic = True) 
        grasp = self.scene_parser.get_grasp(publish_grasp = isPublish, publish_cloud = True)
        # plane = self.scene_parser.get_plane(publish = isPublish)
        plane = None
        
        rospy.loginfo("From pick request to grasp detection took " + str(time.time() - starttime) + " seconds.")
        # ee_pose = self.manipulationMethods.getEndEffectorPose()
        # self.align_to_object_horizontal(ee_pose_x = ee_pose[0], debug_print = {"ee_pose" : ee_pose})

        if grasp:
            grasp_center, grasp_yaw = grasp

            if plane:
                plane_height = plane[1]
                self.heightOfObject = abs(grasp_center[2] - plane_height)
            else:
                self.heightOfObject = 0.2 #default height of object if plane is not detected
            
            offsets = self.offset_dict[self.label2name[objectId]]
            grasp = (grasp_center + np.array(offsets)), grasp_yaw
            
            if use_planner:
                manip_method = self.manipulationMethods.plan_and_pick
            else:                
                manip_method = self.manipulationMethods.pick
            
            manip_method(
                self.trajectory_client, 
                grasp,
                moveUntilContact = self.isContactDict[self.label2name[objectId]]
            ) 
            success = self.scene_parser.is_grasp_success()
            return success
        return False

    def execute_cb(self, goal : PickTriggerGoal):
        starttime = time.time()
        self.goal = goal
        self.scene_parser.set_parse_mode("YOLO", goal.objectId)
        success = self.pick(goal.objectId)
        rospy.loginfo("Pick action took " + str(time.time() - starttime) + " seconds.")
        result = PickTriggerResult()
        result.success = success
        if success:
            self.server.set_succeeded(result)
        else:
            self.server.set_aborted(result)