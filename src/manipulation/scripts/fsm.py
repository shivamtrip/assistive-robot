#! /usr/bin/env python3

from manip_basic_control import ManipulationMethods
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from scene_parser_refactor import SceneParser
from planner.planner import Planner

from drawer_manip import DrawerManager
from pick import PickManager
from place import PlaceManager
from stow import StowManager

class ManipulationManager:

    def __init__(self, loglevel = rospy.INFO):
        rospy.init_node('manipulation', anonymous=True, log_level=loglevel)

        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        rospy.loginfo( "Waiting for trajectory client server...")
        self.trajectoryClient.wait_for_server()

        self.planner = Planner()
        self.scene_parser = SceneParser()
        self.manipulationMethods = ManipulationMethods(self.trajectoryClient)
        
        self.stow_manager   = StowManager(self.scene_parser, self.trajectoryClient, self.manipulationMethods)
        self.drawer_manager = DrawerManager(self.scene_parser, self.trajectoryClient, self.manipulationMethods)
        self.pick_manager   = PickManager(self.scene_parser, self.trajectoryClient, self.manipulationMethods)
        self.place_manager  = PlaceManager(self.scene_parser, self.trajectoryClient, self.manipulationMethods)
        rospy.loginfo( "Manipulation subsystem up.")
    
if __name__ == '__main__':
    node = ManipulationManager()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

