import numpy as np
import rospy
import tf
# from scene_parser import SceneParser
from scene_parser_refactor import SceneParser
from visual_servoing import AlignToObject
from manip_basic_control import ManipulationMethods
import actionlib
from manipulation.msg import DrawerTriggerAction, DrawerTriggerActionGoal, DrawerTriggerActionResult
from helpers import move_to_pose
import time
from enum import Enum
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
        
        self.vs_range = rospy.get_param('/manipulation/vs_range') 
        self.vs_range[0] *= np.pi/180
        self.vs_range[1] *= np.pi/180
        
        
        self.pid_gains = rospy.get_param('/manipulation/pid_gains')
        self.head_tilt_angle_search = rospy.get_param('/manipulation/head_tilt_angle_search_drawer') 
        self.head_tilt_angle_grasp = rospy.get_param('/manipulation/head_tilt_angle_grasp')
        
        self.recoveries = rospy.get_param('/manipulation/recoveries')
        
        self.forward_vel = rospy.get_param('/manipulation/forward_velocity')
        self.interval_before_restart = rospy.get_param('/manipulation/forward_velocity')
        self.max_graspable_distance = rospy.get_param('/manipulation/max_graspable_distance')
        
        self.offset_dict = rospy.get_param('/manipulation/offsets')
        self.isContactDict = rospy.get_param('/manipulation/contact')
        
        self.class_list = rospy.get_param('/object_detection/class_list')
        self.label2name = {i : self.class_list[i] for i in range(len(self.class_list))}

        rospy.loginfo("Loaded pick manager.")    
    
    
    def find_and_align_to_object(self):
        """
        as soon as the aruco is found, reorient and align to it instantly.
        """
        
        rospy.loginfo(f"[{self.node_name}]: Finding and orienting to Drawer")
        move_to_pose(self.trajectory_client, {
            'head_tilt;to' : - self.head_tilt_angle_search * np.pi/180,
            'head_pan;to' : self.vs_range[0],
        })
        rospy.sleep(2)
        nRotates = 0
        while not rospy.is_shutdown():
            for i, angle in enumerate(np.linspace(self.vs_range[0], self.vs_range[1], self.vs_range[2])):
                move_to_pose(self.trajectory_client, {
                    'head_pan;to' : angle,
                })
                rospy.sleep(self.vs_range[3]) #settling time.
                (x, y, z, r, p, angleToGo, confidence, pred_time), isLive = self.scene_parser.get_latest_observation()
                if isLive:
                    drawer_loc, success = self.realign_base_to_aruco(offset = -np.pi/2)
                    if success:
                        move_to_pose(self.trajectory_client, {
                            'head_pan;to' : 0,
                        })
                        return True

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
        
        return False

    def align_to_object_horizontal(self, ee_pose_x = 0, debug_print = {}):

        if not self.scene_parser.clearAndWaitForNewObject(2):
            return False

        xerr = np.inf
        rospy.loginfo("Aligning object horizontally")
        
        prevxerr = np.inf
        moving_avg_n = self.moving_avg_n
        moving_avg_err = []
        mean_err = np.inf
        
        kp = self.pid_gains['kp']
        kd = self.pid_gains['kd']
        curvel = 0.0
        prevsettime = time.time()
        while len(moving_avg_err) <= moving_avg_n and  mean_err > self.mean_err_horiz_alignment:
            [x, y, z, r, p, y, conf, pred_time], isLive = self.scene_parser.get_latest_observation()
            # if isLive:
            rospy.loginfo("Time since last detection = {}".format(time.time() - pred_time))
            xerr = (x - ee_pose_x)
            
            moving_avg_err.append(xerr)
            if len(moving_avg_err) > moving_avg_n:
                moving_avg_err.pop(0)
                
            dxerr = (xerr - prevxerr)
            vx = (kp * xerr + kd * dxerr)
            prevxerr = xerr
            
            mean_err = np.mean(np.abs(np.array(moving_avg_err)))
            debug_print["xerr"] = xerr
            debug_print["mean_err"] = mean_err
            debug_print["x, y, z"] = [x, y, z]
            
            print(debug_print)

            move_to_pose(self.trajectoryClient, {
                    'base_translate;vel' : vx,
            }, asynchronous= True) 
            curvel = vx
            prevsettime = time.time()
            rospy.sleep(0.1)
        
        move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : 0.0,
        }, asynchronous= True) 
        
        return True

    def pick(self, objectId, use_planner = False, isPublish = True):
        self.scene_parser.set_parse_mode("YOLO", objectId)
        rospy.loginfo("Picking object" + str(objectId))
        
        self.manipulationMethods.move_to_pregrasp(self.trajectory_client)
        self.scene_parser.set_point_cloud(publish = isPublish, use_detic = True) 
        grasp = self.scene_parser.get_grasp(publish = isPublish)
        plane = self.scene_parser.get_plane(publish = isPublish)
        
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
