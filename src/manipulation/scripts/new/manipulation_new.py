import numpy as np
import rospy
from std_srvs.srv import Trigger
from helpers import *
from control_msgs.msg import FollowJointTrajectoryAction
from planner import Planner
class Manipulation:
    def __init__(self):
        self.node_name = "manipulation"
        rospy.init_node(self.node_name, anonymous=True)
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        self.stow_robot_service.wait_for_service()
        
        self.navigate_to_location_service = rospy.ServiceProxy('/navigate_to_location', Trigger)
        self.navigate_to_location_service.wait_for_service()
        
        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.search_range = rospy.get_param('/manipulation/search_range')

        self.planner = Planner()
    
        
    def search_for_object(self, accumulate_tsdf = False):
        
        rospy.loginfo(f"[{self.node_name}]: Searching for object")
        
        if not accumulate_tsdf:
            self.enable_tracking()
        else:
            self.enable_tsdf_accumulation()
        
        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -10 * np.pi/180,
            'head_pan;to' : self.search_range[0][0],
        }, asynchronous= False)
        
        nRotates = 0
        for i, angle in enumerate(np.arange(self.search_range[0][0], self.search_range[0][1], self.search_range[1])):
            move_to_pose(self.trajectoryClient, {
                'head_pan;to' : angle,
            }, asynchronous= False)
            if not accumulate_tsdf:
                self.capture_snapshot_request()
            
        if not accumulate_tsdf:
            self.object_of_interest = self.disable_tracking() # returns the initial action server call 
        
        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : -10 * np.pi/180,
            'head_pan;to' : 0,
        }, asynchronous= False)
        
    def move_to_object(self, obj):
        x_me, y_me = self.get_robot_position()
        x_obj, y_obj, z_obj = obj
        
        theta = np.arctan2((y_obj - y_me), (x_obj - x_me))
        
        reachable = 0.85
        location_to_go = [
            x_obj - reachable * np.cos(theta),
            y_obj - reachable * np.sin(theta),
        ]
        
        # send request to movebase
        
    def execute_plan(self, plan):
        pass
    
    def main(self):
        self.search_for_object()
        self.move_to_object(self.object_of_interest)
        self.search_for_object(True)
        plan = self.planner.plan()
        
        if plan != None:
            self.execute_plan()
        else:
            rospy.loginfo(f"[{self.node_name}]: No feasible plan found. Reroute and reattempt.")
