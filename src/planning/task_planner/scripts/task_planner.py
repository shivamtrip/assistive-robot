#!/usr/local/lib/robot_env/bin/python3

#
# Copyright (C) 2023 Auxilio Robotics
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

# ROS imports
import actionlib
from helpers import move_to_pose
import rospkg
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from control_msgs.msg import FollowJointTrajectoryAction
from geometry_msgs.msg import PoseWithCovarianceStamped
from alfred_msgs.msg import Speech, SpeechTrigger
from alfred_msgs.srv import GlobalTask, GlobalTaskResponse, VerbalResponse, VerbalResponseRequest, GlobalTaskRequest, UpdateParam, UpdateParamRequest, UpdateParamResponse
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult, MoveBaseResult
from manipulation.msg import TriggerAction, TriggerFeedback, TriggerResult, TriggerGoal
from alfred_navigation.msg import NavManAction, NavManGoal
from state_manager import BotStateManager, Emotions, LocationOfInterest, GlobalStates, VerbalResponseStates, OperationModes
import os
import json
from enum import Enum

import numpy as np

from utils import get_quaternion

class TaskPlanner:
    def __init__(self):
        rospy.init_node('task_planner')

        self.wakeword_topic_name = rospy.get_param("wakeword_topic_name",
                             "/interface/wakeword_detector/wakeword")
        
        rospack = rospkg.RosPack()
        base_dir = rospack.get_path('task_planner')
        locations_file = rospy.get_param("locations_file", "config/locations.json")
        locations_path = os.path.join(base_dir, locations_file)

        self.goal_locations = json.load(open(locations_path))
        self.startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        self.startNavService = rospy.ServiceProxy('/switch_to_navigation_mode', Trigger)

        self.navigation_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.trigger_initial_pose_pub = False
        self.navigation_client_old = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.manipulation_client = actionlib.SimpleActionClient('manipulation_fsm', TriggerAction)

        self.verbal_response_service_name = rospy.get_param("verbal_response_service_name",
                                "/interface/response_generator/verbal_response_service")

        self.startedListeningService = rospy.Service('/startedListening', Trigger, self.wakeWordTriggered)
        self.commandReceived = rospy.Service('/robot_task_command', GlobalTask, self.command_callback)
        
        self.heightOfObject = 0.84 # change this to a local variable later on 
 
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for stow robot service...")
        self.stow_robot_service.wait_for_service()

        self.verbal_response_service = rospy.ServiceProxy(self.verbal_response_service_name, VerbalResponse)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for verbal response service...")
        self.verbal_response_service.wait_for_service()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for move_base server...")
        self.navigation_client_old.wait_for_server()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for manipulation_fsm server...")
        self.manipulation_client.wait_for_server()

        self.bot_state = BotStateManager(self.stateUpdateCallback)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Bot state manager initialized")


        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for driver..")
        self.trajectoryClient.wait_for_server()

        # Create a service to update firebase params
        self.updateParamService = rospy.Service('/update_param', UpdateParam, self.updateParamServiceCallback)

        self.task_requested = False
        self.time_since_last_command = rospy.Time.now()
        self.startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        self.startManipService.wait_for_service()
        self.isManipMode = False
        self.runningTeleop = False

        self.class_list = rospy.get_param("/object_detection/class_list")
# class_list:
#   # - 'drawer'
#   - 'soda_can' # 0
#   - 'tissue_paper' #1
#   - 'toothbrush' #2
#   - 'tie' #3
#   - 'cell phone' #4
#   - 'banana' #5
#   - 'apple' #6
#   - 'orange' #7
#   - 'bottle' #8
#   - 'cup' #9
#   - 'teddy bear' #10
#   - 'remote' #11
        self.mappings = {}
        self.location_object_map = {
            "teddy bear" : LocationOfInterest.LIVING_ROOM,
            "tie" : LocationOfInterest.LIVING_ROOM,
            "cell phone" : LocationOfInterest.LIVING_ROOM,
            "toothbrush" : LocationOfInterest.LIVING_ROOM,

            "tissue_paper" : LocationOfInterest.ROBOTS,
            "remote" : LocationOfInterest.ROBOTS,

            "cup" : LocationOfInterest.KITCHEN,
            "soda_can" : LocationOfInterest.KITCHEN,
            "apple" : LocationOfInterest.KITCHEN,
            "bottle" : LocationOfInterest.KITCHEN,
            "orange" : LocationOfInterest.KITCHEN,
            "banana" : LocationOfInterest.KITCHEN,
        }
        
        for i, cls in enumerate(self.class_list):
            self.mappings[cls] = [self.location_object_map[cls], i]    
           
        
            
        self.mappings['do_nothing'] = [LocationOfInterest.LIVING_ROOM, -1]
        
        self.navigationGoal = LocationOfInterest.HOME
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Node Ready.")

        self.navigation_client = actionlib.SimpleActionClient('nav_man', NavManAction)

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for Navigation Manager server...")
        self.navigation_client.wait_for_server()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Subscribing to the amcl pose topic...")
        self.amcl_initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Node Ready.")
        # create a timed callback to check if we have, Obj been idle for too long

    def updateParamImpl(self, path, value):
        self.bot_state.update_param(path, value)

    def updateParamServiceCallback(self, msg):
        self.updateParamImpl(msg.path, msg.value)
        return UpdateParamResponse()

    def idleCallback(self, event):
        if not self.bot_state.isAutonomous() and self.bot_state.isExtendedCommunicationLossTeleop():
            self.bot_state.currentGlobalState = GlobalStates.IDLE

    def stateUpdateCallback(self):
        
        if not self.bot_state.isAutonomous():
            if not self.isManipMode :
                self.isManipMode = True
                self.startManipService()
            
            if not self.runningTeleop:
                self.maintainvelocities()

    def maintainvelocities(self):
        self.runningTeleop = True
        while not rospy.is_shutdown() and self.bot_state.isAutonomous() == False:
            commands = self.bot_state.get_teleop_commands_from_firebase(None, callback = False)
            rospy.loginfo(f"[{rospy.get_name()}]:" + "Received teleop commands: {}".format(commands))
            if commands['mobile_base']['velx'] == 0:
                move_to_pose(self.trajectoryClient, 
                    {'base_rotate|;vel' : commands['mobile_base']['veltheta']*2},
                    asynchronous= True
                )
            else:
                move_to_pose(self.trajectoryClient, 
                    {'base_translate|;vel' : commands['mobile_base']['velx']/2},
                    asynchronous= True
                )
            gripperpos = -50
            if commands['manipulator']['gripper_open']:
                gripperpos = 100
            move_to_pose(self.trajectoryClient, 
                {
                    'stretch_gripper|;to' : gripperpos,
                    'wrist_yaw|;to' : commands['manipulator']['yaw_position'] * np.pi/180,
                    'lift|;vel' : commands['manipulator']['vel_lift'],
                    'arm|;vel' : -commands['manipulator']['vel_extend']
                },
                asynchronous= True
            )
            
            rospy.sleep(0.1)

        self.runningTeleop = False
        self.stow_robot_service()
        print("teleop stopped")

    def executeTask(self):

        if not self.bot_state.isAutonomous():
            rospy.loginfo("Received a command, but not in autonomous mode, so ignoring. Ask to relenquish control.")
            return

        rospy.loginfo("Task has been requested")
        rospy.loginfo("Sending verbal response to acknowledge task reception.")
        self.process_verbal_response(VerbalResponseStates.ON_IT)

        self.stow_robot_service()
        self.startNavService()
        rospy.sleep(0.5)
        rospy.loginfo("Going to {}, to manipulate {}".format(self.navigationGoal.name, self.objectOfInterest))

        # self.upda
        self.updateParamImpl("current_task/object_of_interest", self.objectOfInterest)
        self.updateParamImpl("visualization/task_name", "go_to_object")
        
        navSuccess = self.navigate_to_location_navman(self.navigationGoal)
        # navSuccess = self.navigate_to_location(self.navigationGoal)
        # if not navSuccess:
        #     rospy.loginfo("Failed to navigate to table, adding intermediate waypoints")
        #     navSuccess = self.navigate_to_location(self.navigationGoal)
        
        self.startManipService()
        rospy.sleep(0.5)

        self.bot_state.currentGlobalState = GlobalStates.MANIPULATION
        rospy.loginfo("Attempting to find and manipulate objects.")
        self.updateParamImpl("visualization/task_name", "pick_object")
        manipulationSuccess = self.manipulate_object(self.objectOfInterest, isPick = True)
        # if not manipulationSuccess:
        #     rospy.loginfo("Manipulation failed. Cannot find object, returning to user")
        
        self.stow_robot_service()
        rospy.sleep(2)
        self.trigger_initial_pose_pub = True
        rospy.sleep(2)

        self.startNavService()
        rospy.sleep(0.5)                                                        
        # rospy.loginfo(f"Going to table. Manipulation success = {manipulationSuccess}")
        # self.updateParamImpl("visualization/task_name", "go_to_user")
        success = self.navigate_to_location_navman(LocationOfInterest.TABLE)
        # success = self.navigate_to_location(self.navigationGoal)
        # if not success:
        #     rospy.loginfo("Failed to navigate to table, adding intermediate waypoints")
        #     self.navigate_to_location_navman(LocationOfInterest.TABLE)

        self.bot_state.currentGlobalState = GlobalStates.MANIPULATION
        self.startManipService()
        manipulationSuccess = False
        if manipulationSuccess:
            rospy.loginfo("Requesting to place the object")
            self.updateParamImpl("visualization/task_name", "place_object")
            self.manipulate_object(self.objectOfInterest, isPick = False)
            self.process_verbal_response(VerbalResponseStates.HERE_YOU_GO)
        else:
            rospy.loginfo("Requesting verbal command.")
            self.updateParamImpl("visualization/task_name", "verbal_command")
            self.process_verbal_response(VerbalResponseStates.SORRY)

        self.bot_state.update_param("current_task/object_of_interest", "")

        self.bot_state.currentGlobalState = GlobalStates.IDLE
        self.task_requested = False
        self.time_since_last_command = rospy.Time.now()


    def manipulate_object(self, object : int, isPick = True):
        rospy.loginfo("Starting manipulation, going to perform object" + (" picking" if isPick else " placing"))
        rospy.loginfo("Manipulating {}".format(object))
        self.bot_state.currentGlobalState = GlobalStates.MANIPULATION
        # send goal
        goalObject = object
        # if not isPick:
        #     goalObject = 60 #table

        goal = TriggerGoal(isPick = isPick, heightOfObject = self.heightOfObject)
        goal.objectId = goalObject
        self.manipulation_client.send_goal(goal, feedback_cb = self.bot_state.manipulation_feedback)
        self.manipulation_client.wait_for_result()
        result: TriggerResult = self.manipulation_client.get_result()
        if result.success:
            self.heightOfObject = result.heightOfObject
            rospy.loginfo("Manipulation complete")
        return result.success
    
    def process_verbal_response(self, response):
        msg = VerbalResponseRequest()
        if response == VerbalResponseStates.HERE_YOU_GO:
            msg.response = "here_you_go"
            res = self.verbal_response_service(msg)
            self.bot_state.update_emotion(Emotions.HAPPY)
        elif response == VerbalResponseStates.ON_IT:
            msg.response = "on_it"
            res = self.verbal_response_service(msg)
        elif response == VerbalResponseStates.SORRY:
            msg.response = "Sorry. I could not find your object."
            res = self.verbal_response_service(msg)


    def wakeWordTriggered(self, msg):
        self.bot_state.update_hri("wake_word", Emotions.NEUTRAL) 
        rospy.loginfo("Wake word triggered")
        return TriggerResponse()
        
    def command_callback(self, msg : GlobalTaskRequest):
        self.bot_state.currentGlobalState = GlobalStates.RECEIVED_COMMAND
        rospy.loginfo("Received input")
        self.bot_state.update_hri(msg.speech, Emotions.NEUTRAL)

        if msg.primitive == 'video_call':
            rospy.loginfo("Updating operation mode to video call")
            self.bot_state.update_operation_mode(OperationModes.TELEOPERATION)
        elif msg.primitive == 'stop_video':
            rospy.loginfo("Updating operation mode to autonomous")
            self.bot_state.update_operation_mode(OperationModes.AUTONOMOUS)
        elif msg.primitive != 'do_nothing':
            self.navigationGoal, self.objectOfInterest = self.mappings[msg.primitive]
            self.executeTask()

        return GlobalTaskResponse(success = True)
    
    def amcl_pose_callback(self, msg : PoseWithCovarianceStamped):
        """ takes the amcl pose and republishes it to /initialpose whenever required."""

        if(self.trigger_initial_pose_pub):

            # Get the covariance matrix and set the covariance of x and y to 0.06
            pose = msg.pose.pose
            cov = np.array(msg.pose.covariance).reshape(6,6)
            rospy.loginfo("Received amcl pose: {}".format(pose))
            rospy.loginfo("Received amcl covariance: {}".format(cov))
            cov[0:2, 0:2] = np.eye(2)*0.06
            cov_flat = cov.flatten()

            # Generate pub msg
            pub_msg = PoseWithCovarianceStamped()
            pub_msg.header = msg.header
            pub_msg.pose.pose = pose
            pub_msg.pose.covariance = cov_flat.tolist()

            self.amcl_initial_pose_pub.publish(pub_msg)
            self.trigger_initial_pose_pub = False
            rospy.loginfo("Published initial pose after the manipulation ends!!")
            # rospy.loginfo("Published (pose, cov)=", (pub_msg.pose.pose,pub_msg.pose.covariance))
    
    def navigate_to_location_navman(self, location : Enum):
        locationName = location.name
        goal = NavManGoal()
        goal.x = self.goal_locations[locationName]['x']
        goal.y = self.goal_locations[locationName]['y']
        goal.theta = self.goal_locations[locationName]['theta']
        
        self.navigation_client.send_goal(goal, feedback_cb = self.bot_state.navigation_feedback)
        wait = self.navigation_client.wait_for_result()

        if self.navigation_client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Failed to reach {}".format(locationName))
            self.navigation_client.cancel_goal()
            return False
    
        return True

    def navigate_to_location(self, location : Enum):
        locationName = location.name
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Executing task. Going to {}".format(locationName))
        # send goal
        print("Sending goal to move_base", self.goal_locations[locationName]['x'], self.goal_locations[locationName]['y'], self.goal_locations[locationName]['theta'])
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose.position.x = self.goal_locations[locationName]['x']
        goal.target_pose.pose.position.y = self.goal_locations[locationName]['y']
        quaternion = get_quaternion(self.goal_locations[locationName]['theta'])
        goal.target_pose.pose.orientation = quaternion
        self.navigation_client_old.send_goal(goal)
        wait = self.navigation_client_old.wait_for_result()

        if self.navigation_client_old.get_state() != actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Failed to reach {}".format(locationName))
            # cancel navigation
            self.navigation_client_old.cancel_goal()
            return False
        
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Reached {}".format(locationName))
        self.bot_state.update_emotion(Emotions.HAPPY)
        self.bot_state.update_state()
        self.bot_state.currentGlobalState = GlobalStates.REACHED_GOAL
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Reached {}".format(locationName))
        return True


if __name__ == "__main__":
    task_planner = TaskPlanner()

    rospy.spin()
    

