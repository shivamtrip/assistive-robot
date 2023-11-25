#!/usr/bin/env python3

from control_msgs.msg import FollowJointTrajectoryGoal
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from helpers import move_to_pose
from std_msgs.msg import String, Header
import time
import json
import numpy as np
from sensor_msgs.msg import JointState, CameraInfo, Image
from enum import Enum
# from helpers import armclient, headclient, diffdrive, gripperclient
import tf
import tf.transformations
from tf import TransformerROS
from tf import TransformListener, TransformBroadcaster
from cv_bridge import CvBridge, CvBridgeError
from helpers import convertPointToDepth
import cv2
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2, PointField
import pandas as pd
# ROS action finite state machine
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryResult
import actionlib
from yolo.msg import Detections
from scene_parser import SceneParser
from alfred_navigation.msg import NavManAction, NavManGoal


class State(Enum):
    SEARCH = 1
    
    MOVE_TO_OBJECT = 3
    ROTATE_90 = 4
    ALIGN_HORIZONTAL = 5
    COMPLETE = 6

    # substates
    SCANNING = 11      
    ALIGN_TO_OBJECT = 12          

    FAILED = -1


class AlignToObjectAdvanced:
    def __init__(self, scene_parser : SceneParser, planner):
        self.node_name = 'visual_alignment'
        self.state = State.SEARCH
        
        self.planner = planner
        
        self.scene_parser = scene_parser
        self.pid_gains = rospy.get_param('/manipulation/pid_gains')
        
        self.prevxerr = 0    
        
        self.recoveries = rospy.get_param('/manipulation/recoveries')
        self.mean_err_horiz_alignment = rospy.get_param('/manipulation/mean_err_horiz_alignment')
        self.vs_range = rospy.get_param('/manipulation/vs_range') 
        self.vs_range[0] *= np.pi/180
        self.vs_range[1] *= np.pi/180
        
        self.moving_avg_n = rospy.get_param('/manipulation/moving_average_n')
        self.max_graspable_distance = rospy.get_param('/manipulation/max_graspable_distance')
        
        self.head_tilt_angle_search = rospy.get_param('/manipulation/head_tilt_angle_search')
        self.head_tilt_angle_grasp = rospy.get_param('/manipulation/head_tilt_angle_grasp')
        
        rospy.loginfo(f"[{self.node_name}]: Waiting for trajectory server")
        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.trajectoryClient.wait_for_server()
        
        # self.navigation_client = actionlib.SimpleActionClient('nav_man', NavManAction)
        # rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for Navigation Manager server...")
        # self.navigation_client.wait_for_server()
        
        self.switch_to_navigation_mode = rospy.ServiceProxy('/switch_to_navigation_mode', Trigger)
        self.switch_to_navigation_mode.wait_for_service()
        
        self.switch_to_manipulation_mode = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        self.switch_to_manipulation_mode.wait_for_service()
        
        self.forward_vel = rospy.get_param('/manipulation/forward_velocity')
        self.interval_before_restart = rospy.get_param('/manipulation/forward_velocity')
        
        self.obj_tf_publisher = tf.TransformBroadcaster()
        
        rospy.loginfo(f"[{self.node_name}]: Initial images obtained.")
        rospy.loginfo(f"[{self.node_name}]: Node initialized")
        self.prevtime = time.time()
        
    def is_object_reachable(self, object_location, cloud):
        nav_target, is_required = self.planner.plan_base(cloud, object_location)
        return nav_target, is_required
    
    
    def find_renav_location(self):
        rospy.loginfo(f"[{self.node_name}]: Finding and orienting to object")
        
        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : - self.head_tilt_angle_search * np.pi/180,
            'head_pan;to' : self.vs_range[0],
        })
        
        rospy.sleep(2)
        # self.scene_parser.reset_tsdf()
        
        settling_time = self.vs_range[3]
        while not rospy.is_shutdown():
            self.scene_parser.clear_observations(wait = True)
            for i, angle in enumerate(np.linspace(self.vs_range[0], self.vs_range[1], self.vs_range[2])):
                move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : angle,
                })
                rospy.sleep(settling_time) #settling time.
                self.scene_parser.capture_shot()
            
            [angleToGo, x, y, z, radius], success = self.scene_parser.estimate_object_location()
            self.objectLocation = [x, y, z]
            move_to_pose(self.trajectoryClient, {   
                'head_pan;to' : angleToGo,
            })
            
            success = self.scene_parser.clearAndWaitForNewObject(10)

            if not success:
                settling_time += 1
                continue
            
            self.scene_parser.capture_shot()
            
            [angleToGo, x, y, z, radius], success = self.scene_parser.estimate_object_location()
            self.objectLocation = [x, y, z]
            
            cloud = self.scene_parser.get_pcd_from_tsdf(publish = True)
            # cloud = None
            return self.objectLocation, cloud, success
            

    def alignObjectForManipulation(self):
        """
        Rotates base to align manipulator for grasping
        """
        rospy.loginfo(f"[{self.node_name}]: Aligning object for manipulation")
        [angleToGo, x, y, z, radius], success = self.scene_parser.estimate_object_location()
        move_to_pose(self.trajectoryClient, {
                'head_pan;to' : -np.pi/2,
                'base_rotate;by' : angleToGo + np.pi/2,
            }
        )
        rospy.sleep(4)
        return True
    
         
    def main(self):
        rospy.loginfo("Triggered Visual Servoing")
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
                    self.alignObjectForManipulation()
                    return True
                    
                target = object_location
                theta = np.arctan2(target[1], target[0])

                move_to_pose(self.trajectoryClient, {
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

if __name__ == "__main__":
    # unit test
    switch_to_manipulation = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    switch_to_manipulation.wait_for_service()
    switch_to_manipulation()
    rospy.init_node("align_to_object", anonymous=True)
    node = AlignToObjectAdvanced(8)
    # node.main(8)
    node.findAndOrientToObject()
    # node.moveTowardsObject()
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass