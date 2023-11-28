import numpy as np
import rospy
import tf
# from scene_parser import SceneParser
from scene_parser_refactor import SceneParser
from visual_servoing import AlignToObject
from manip_basic_control import ManipulationMethods
import actionlib
from manipulation.msg import FindAlignAction, FindAlignResult, FindAlignGoal
from std_srvs.srv import Trigger, TriggerResponse

from helpers import move_to_pose
import time
import actionlib
import open3d as o3d

class FindAndAlignManager():
    def __init__(self, scene_parser : SceneParser, trajectory_client, manipulation_methods : ManipulationMethods, planner):
        self.node_name = 'pick_manager'
        self.scene_parser = scene_parser
        self.trajectory_client = trajectory_client
        self.manipulationMethods = manipulation_methods
        self.planner = planner
        
        self.server = actionlib.SimpleActionServer('find_align_action', FindAlignAction, execute_cb=self.execute_cb, auto_start=False)
        
        self.switch_to_navigation_mode = rospy.ServiceProxy('/switch_to_navigation_mode', Trigger)
        self.switch_to_navigation_mode.wait_for_service()
        
        self.switch_to_manipulation_mode = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        self.switch_to_manipulation_mode.wait_for_service()
        
        self.vs_range = rospy.get_param('/manipulation/vs_range') 
        self.vs_range[0] *= np.pi/180
        self.vs_range[1] *= np.pi/180
        
        self.recoveries = rospy.get_param('/manipulation/recoveries')

        self.offset_dict = rospy.get_param('/manipulation/offsets')
        self.isContactDict = rospy.get_param('/manipulation/contact')
        
        self.moving_avg_n = rospy.get_param('/manipulation/moving_average_n')
        self.max_graspable_distance = rospy.get_param('/manipulation/max_graspable_distance')
        
        self.head_tilt_angle_search = rospy.get_param('/manipulation/head_tilt_angle_search')
        self.head_tilt_angle_grasp = rospy.get_param('/manipulation/head_tilt_angle_grasp')
        
        self.class_list = rospy.get_param('/object_detection/class_list')
        self.label2name = {i : self.class_list[i] for i in range(len(self.class_list))}
        self.server.start()
        
        rospy.loginfo("Loaded FindAndAlignManager")    
    
    def find_renav_location(self):
        rospy.loginfo(f"Finding and orienting to object")
        
        move_to_pose(self.trajectory_client, {
            'head_tilt;to' : - self.head_tilt_angle_search * np.pi/180,
            'head_pan;to' : self.vs_range[0],
        })
        
        rospy.sleep(2)
        
        settling_time = self.vs_range[3]
        
        nRotates = 0
        
        while not rospy.is_shutdown():
            self.scene_parser.clear_observations(wait = True)
            for i, angle in enumerate(np.linspace(self.vs_range[0], self.vs_range[1], self.vs_range[2])):
                move_to_pose(self.trajectory_client, {
                    'head_pan;to' : angle,
                })
                rospy.sleep(settling_time) #settling time.
                self.scene_parser.capture_shot()
            
            objectProperties, success = self.scene_parser.estimate_object_location()
            if success:
                angle = objectProperties['angle']
                object_location = [
                    objectProperties['x'],
                    objectProperties['y'],
                    objectProperties['z'],
                    objectProperties['angle']
                ]
                cloud = self.scene_parser.get_pcd_from_tsdf(publish = True)
                return object_location, cloud, success
            
            rospy.loginfo("Object not found, rotating base.")
            move_to_pose(self.trajectory_client, {
                'base_rotate;by' : -np.pi/2,
                'head_pan;to' : self.vs_range[0],
            })
            rospy.sleep(5)
            
            self.scene_parser.clear_observations()
            nRotates += 1
            if nRotates >= self.recoveries['n_scan_attempts']:
                return False
            
    def align_to_object_for_manipulation(self, object_location):
        x, y, z, theta = object_location
        move_to_pose(self.trajectory_client, {
            'head_tilt;to' : - self.head_tilt_angle_grasp * np.pi/180,
            'head_pan;to' : -np.pi/2,
            'base_rotate;by' : theta + np.pi/2,
        })
        
        self.scene_parser.get_view_frustum()
        
    
    def is_object_reachable(self, object_location, cloud):
        nav_target, is_required = self.planner.plan_base(cloud, object_location)
        return nav_target, is_required
    
    def main(self):
        is_object_found = False
        while not is_object_found:
            self.switch_to_manipulation_mode()
            rospy.sleep(1)
            object_location, cloud, is_object_found = self.find_renav_location()
            rospy.loginfo("Object found: " + str(is_object_found))
            if is_object_found:
                target, is_nav_required = self.is_object_reachable(object_location, cloud)
                
                if not is_nav_required:
                    rospy.loginfo("Object is reachable!")
                    self.switch_to_manipulation_mode()
                    self.alignObjectForManipulation(object_location)
                    return True
                    
                target = object_location
                theta = np.arctan2(target[1], target[0])

                move_to_pose(self.trajectory_client, {
                    'head_tilt;to' : - self.head_tilt_angle_search * np.pi/180,
                    'head_pan;to' : 0,
                })
                # convert target to world frame
                self.scene_parser.publish_point(target, "point_a", frame = 'base_link')
                target = self.scene_parser.convert_point_from_to_frame(target[0], target[1], target[2], from_frame = 'base_link', to_frame = 'map')
                self.scene_parser.publish_point(target, "point_b", frame = 'map')
                self.switch_to_navigation_mode()
                rospy.loginfo("new_to?: " + str(target))
                
                self.navigation_client.send_goal(NavManGoal(
                    x = target[0],
                    y = target[1],
                    theta = theta
                ))
                wait = self.navigation_client.wait_for_result()
            else:
                # add behavior to recover from not finding the object.
                # for now return false
                # request new search location from task_planner.
                return False
        return True
    def execute_cb(self, goal):
        rospy.loginfo("\n\n--------RECEIVED FIND AND ALIGN GOAL--------")
        starttime = time.time()
        self.goal = goal
        self.scene_parser.set_parse_mode("YOLO", goal.objectId)
        success = self.find_renav_location()
        rospy.loginfo("Find and renavigate action took " + str(time.time() - starttime) + " seconds.")
        result = FindAlignResult()
        result.success = success
        if success:
            self.server.set_succeeded(result)
        else:
            self.server.set_aborted(result)