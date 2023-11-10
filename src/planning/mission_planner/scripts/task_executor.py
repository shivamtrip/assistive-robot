#!/usr/local/lib/robot_env/bin/python3


import rospy
import actionlib
from std_srvs.srv import Trigger, TriggerResponse
from alfred_navigation.msg import NavManAction, NavManGoal
import rospkg
import os
import json


class TaskExecutor:


    def __init__(self):


        rospack = rospkg.RosPack()
        base_dir = rospack.get_path('mission_planner')
        locations_file = rospy.get_param("locations_file", "config/map_locations.json")
        locations_path = os.path.join(base_dir, locations_file)

        self.goal_locations = json.load(open(locations_path))



        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for stow robot service...")



        self.navigation_client = actionlib.SimpleActionClient('nav_man', NavManAction)

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for Navigation Manager server...")
        self.navigation_client.wait_for_server()








    def navigate_to_location(self, location_name : str):

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Executing task. Going to {}".format(location_name))
 
        goal = NavManGoal()

        goal.x = self.goal_locations[location_name]['x']
        goal.y = self.goal_locations[location_name]['y']
        goal.theta = self.goal_locations[location_name]['theta']

        self.navigation_client.send_goal(goal, feedback_cb = self.dummy_cb)

        print("Task Executor has sent goal to Navigation Manager")

        wait = self.navigation_client.wait_for_result()

        if self.navigation_client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Failed to reach {}".format(locationName))
            # cancel navigation
            # self.navigation_client.cancel_goal()
            return False
        
        # rospy.loginfo(f"[{rospy.get_name()}]:" +"Reached {}".format(locationName))
        # self.bot_state.update_emotion(Emotions.HAPPY)
        # self.bot_state.update_state()
        # self.bot_state.currentGlobalState = GlobalStates.REACHED_GOAL
        # rospy.loginfo(f"[{rospy.get_name()}]:" +"Reached {}".format(locationName))
        return True



    def dummy_cb(self, msg):
        
        return

