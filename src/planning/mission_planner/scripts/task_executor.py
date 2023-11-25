#!/usr/local/lib/robot_env/bin/python3


import rospy
import actionlib
from std_srvs.srv import Trigger, TriggerResponse
from alfred_navigation.msg import NavManAction, NavManGoal
from manipulation.msg import TriggerAction, TriggerFeedback, TriggerResult, TriggerGoal
import rospkg
import os
import json
# from visual_servoing import AlignToObject


class TaskExecutor:


    def __init__(self):


        rospack = rospkg.RosPack()
        base_dir = rospack.get_path('mission_planner')
        locations_file = rospy.get_param("locations_file", "config/map_locations.json")
        locations_path = os.path.join(base_dir, locations_file)

        self.goal_locations = json.load(open(locations_path))
        # self.visualServoing = AlignToObject(-1)

        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for stow robot service...")
        self.stow_robot_service.wait_for_service()

        self.startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        self.startManipService.wait_for_service()


        self.startNavService = rospy.ServiceProxy('/switch_to_navigation_mode', Trigger)
        self.startNavService.wait_for_service()

        self.navigation_client = actionlib.SimpleActionClient('nav_man', NavManAction)
        self.manipulation_client = actionlib.SimpleActionClient('manipulation_manager', TriggerAction)
        self.telepresence_client = actionlib.SimpleActionClient('telepresence_manager', TriggerAction)

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for Navigation Manager server...")
        self.navigation_client.wait_for_server()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for Manipulation Manager server...")
        self.manipulation_client.wait_for_server()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for Telepresence Manager server...")
        self.telepresence_client.wait_for_server()

        self.heightOfObject = 0.84 # change this to a local variable later on 


        self.stow_robot_service()

        self.startNavService()
        rospy.sleep(0.5)


        print("Task Executor Initialized")






    def navigate_to_location(self, location_name : str):

        self.startNavService()
        rospy.sleep(0.5)


        rospy.loginfo(f"[{rospy.get_name()}]:" + "Executing task. Going to {}".format(location_name))
 
        goal = NavManGoal()

        goal.x = self.goal_locations[location_name]['x']
        goal.y = self.goal_locations[location_name]['y']
        goal.theta = self.goal_locations[location_name]['theta']

        self.navigation_client.send_goal(goal, feedback_cb = self.dummy_cb)

        print("Task Executor has sent goal to Navigation Manager")

        wait = self.navigation_client.wait_for_result()

        if self.navigation_client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Failed to reach {}".format(location_name))
            return False
        
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Reached {}".format(location_name))

        return True
    

    

    def manipulate_object(self, object, isPick = True):

        self.startManipService()

        rospy.loginfo("Task Executor starting manipulation, going to perform object" + (" picking" if isPick else " placing"))
        rospy.loginfo("Manipulating {}".format(object.name))
        # send goal
        goalObject = object.value
        if not isPick:
            goalObject = 60 #table

        goal = TriggerGoal()
        goal.isPick = isPick
        goal.objectId = goalObject
        self.manipulation_client.send_goal(goal, feedback_cb = self.dummy_cb)
        print("Task Executor has sent goal to Manipulation Manager")
        self.manipulation_client.wait_for_result()
        print("Task Executor received result from Manipulation Manager")
        result: TriggerResult = self.manipulation_client.get_result()
        if result.success:
            self.heightOfObject = result.heightOfObject
            rospy.loginfo("Manipulation complete")

        if isPick:  
            self.stow_robot_service()

        return result.success




    def align_to_user(self, user_id):
        
        self.startManipService()

        goal = TriggerGoal()
        goal.objectId = user_id
        self.telepresence_client.send_goal(goal, feedback_cb = self.dummy_cb)
        print("Task Executor has sent goal to Telepresence Manager")
        self.manipulation_client.wait_for_result()
        print("Task Executor received result from Telepresence Manager")
        result: TriggerResult = self.manipulation_client.get_result()

        if(result.success):
            rospy.loginfo("Manipulation complete")




    def dummy_cb(self, msg):
        
        return

