#!/usr/local/lib/robot_env/bin/python3


import rospy
import actionlib
from std_srvs.srv import Trigger, TriggerResponse
from alfred_navigation.msg import NavManAction, NavManGoal



class TaskExecutor:


    def __init__(self):


        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for stow robot service...")



        self.navigation_client = actionlib.SimpleActionClient('nav_man', NavManAction)

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for Navigation Manager server...")
        self.navigation_client.wait_for_server()








    def navigate_to_location(self):     # location : Enum):
        # locationName = location.name
        # rospy.loginfo(f"[{rospy.get_name()}]:" +"Executing task. Going to {}".format(locationName))
        # send goal

        print("NAVIGATING TO LOCATION")
        goal = NavManGoal()


        # near AIMS fridge
        goal.x = -2.615337371826172
        goal.y = -7.306527137756348
        goal.theta = 0


        self.navigation_client.send_goal(goal, feedback_cb = self.dummy_cb)

        print("Task Planner has sent goal to Navigation Manager")

        return True



    def dummy_cb(self, msg):
        
        return

        # print("Sending goal to move_base", self.goal_locations[locationName]['x'], self.goal_locations[locationName]['y'], self.goal_locations[locationName]['theta'])
        # goal = MoveBaseGoal()
        # goal.target_pose.header.frame_id = "map"
        # goal.target_pose.pose.position.x = self.goal_locations[locationName]['x']
        # goal.target_pose.pose.position.y = self.goal_locations[locationName]['y']
        # quaternion = get_quaternion(self.goal_locations[locationName]['theta'])
        # goal.target_pose.pose.orientation = quaternion
        # self.navigation_client.send_goal(goal, feedback_cb = self.bot_state.navigation_feedback)
        # wait = self.navigation_client.wait_for_result()

        # if self.navigation_client.get_state() != actionlib.GoalStatus.SUCCEEDED:
        #     rospy.loginfo(f"[{rospy.get_name()}]:" +"Failed to reach {}".format(locationName))
        #     # cancel navigation
        #     self.navigation_client.cancel_goal()
        #     return False
        
        # rospy.loginfo(f"[{rospy.get_name()}]:" +"Reached {}".format(locationName))
        # self.bot_state.update_emotion(Emotions.HAPPY)
        # self.bot_state.update_state()
        # self.bot_state.currentGlobalState = GlobalStates.REACHED_GOAL
        # rospy.loginfo(f"[{rospy.get_name()}]:" +"Reached {}".format(locationName))
        # return True
