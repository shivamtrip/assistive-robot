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
from firebase_node import FirebaseNode
import threading
import concurrent.futures
from visual_servoing import AlignToObject
# import stretch_body.pimu as pimu
from sensor_msgs.msg import BatteryState

# from alfred_msgs.msg import Speech, SpeechTrigger
# from alfred_msgs.srv import GlobalTask, GlobalTaskResponse, VerbalResponse, VerbalResponseRequest, GlobalTaskRequest
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult, MoveBaseResult
# from manipulation.msg import TriggerAction, TriggerFeedback, TriggerResult, TriggerGoal

from state_manager import BotStateManager, Emotions, LocationOfInterest, GlobalStates, ObjectOfInterest, VerbalResponseStates, OperationModes
import os
import json
from enum import Enum
import pyrebase
import time

from utils import get_quaternion

class TaskPlanner:
    def __init__(self):
        self.trigger=0
        # self.q=[]
        self.eta=0
        self.eta2=0
        self.q = []
        self.last_task_success=0
        rospy.init_node('task_planner')
        rospack = rospkg.RosPack()
        base_dir = rospack.get_path('task_planner')
        locations_file = rospy.get_param("locations_file", "config/locations.json")
        locations_path = os.path.join(base_dir, locations_file)
        self.goal_locations = json.load(open(locations_path))
        self.visualServoing = AlignToObject(-1)
        self.bat_sub = rospy.Subscriber('/battery', BatteryState, self.battery_check)
        self.nServoTriesAttempted=0
        self.nServoTriesAllowed=3
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        firebase_secrets_path = os.path.expanduser("~/firebasesecrets.json")
        if not os.path.isfile(firebase_secrets_path):
            raise FileNotFoundError("Firebase secrets file not found")
        
        with open(firebase_secrets_path, 'r') as f:
            config = json.load(f)

        # with open(os.path.expanduser("~/alfred-autonomy/src/planning/task_planner/config/firebase_schema.json")) as f:
        #     self.state_dict = json.load(f)

        self.firebase = pyrebase.initialize_app(config)
        self.db = self.firebase.database()
        # self.db.remove("")
        # self.state_dict["button_callback"] = 0
        # self.state_dict["button_callback2"] = 0
        # self.state_dict["button_callback3"] = 0
        # self.state_dict["button_callback4"] = 0

        # self.db.set(self.state_dict) # fill
 
        self.last_update = time.time()

        self.bot_state = BotStateManager(self.stateUpdateCallback)



        rospack = rospkg.RosPack()
        self.startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        self.startNavService = rospy.ServiceProxy('/switch_to_navigation_mode', Trigger)
        self.startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)


        self.navigation_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        #self.commandReceived = rospy.Service('/robot_task_command', GlobalTask, self.command_callback)
        
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for stow robot service...")
        # self.stow_robot_service.wait_for_service()

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for move_base server...")
        self.navigation_client.wait_for_server()

        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for driver..")
        self.trajectoryClient.wait_for_server()


        self.task_requested = False
        self.time_since_last_command = rospy.Time.now()
        self.startManipService.wait_for_service()

        self.isManipMode = False
        self.runningTeleop = False

        self.mappings = {
            'fetch_water' : [LocationOfInterest.LIVING_ROOM, ObjectOfInterest.BOTTLE],
            'fetch_apple' : [LocationOfInterest.NET, ObjectOfInterest.APPLE],
            'fetch_remote' : [LocationOfInterest.LIVING_ROOM, ObjectOfInterest.REMOTE],
            'do_nothing' : [LocationOfInterest.LIVING_ROOM, ObjectOfInterest.NONE],
            'fetch_banana' : [LocationOfInterest.LIVING_ROOM, ObjectOfInterest.BANANA],
        }
        self.navigationGoal = LocationOfInterest.HOME
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Node Ready.")

        # create a timed callback to check if we have, Obj been idle for too long

    def stateUpdateCallback(self):
        pass        
        # if not self.bot_state.isAutonomous():
        #     if not self.isManipMode :
        #         self.isManipMode = True
        #         self.startManipService()
            
            # if not self.runningTeleop:
                # self.maintainvelocities()
    def delay(self):
        time.sleep(5)

    def main(self):
        task_thread = threading.Thread(target=task_planner.execute_task_temp)
        task_thread2 = threading.Thread(target=task_planner.updateButton)
        task_thread.start()
        task_thread2.start()

        task_thread.join()
        task_thread2.join()

    def execute_task_temp(self):
        
        while True:
            try:
                if len(self.q) != 0:
                    # remote_id, task_to_execute = self.q.pop(0)
                    temp=self.q[0]
                    tasks_to_execute = self.q.pop(0)
                    # execute tasks here.
                    if(temp in [1,2,3,4]):
                        pp=1
                    elif(temp in [5,6,7,8]):
                        pp=2
                    rospy.loginfo(f"Executing task {tasks_to_execute} for room {pp}")
                    self.navigate_to_location(self.navigationGoal, tasks_to_execute)


                    rospy.sleep(5)

                    rospy.loginfo("Task complete...")
            except KeyboardInterrupt:
                break


        # if(len(self.q)<4):
        #     if(len(self.q)!=4):
        #         self.q.append(0)
        #         if(len(self.q)!=4):
        #             self.q.append(0)
        # if((self.q)and(len(self.q)>3)):
        #     # task_thread = threading.Thread(target=self.executeTask, args=(0,self.q[0]))
        #     # self.executeTask(0, self.q[0])
        #     self.trigger=1
        #     if(self.q[0]==1):
        #         rospy.loginfo("Jake requested for waterr.")
        #         self.q.popleft()
        #         # self.last_task_success=self.q[0]
        #         self.delay()
        #     elif(self.q[0]==2):
        #         rospy.loginfo("Jake wants to talk to Donna.")
        #         self.q.popleft()
        #         time.sleep(5)
        #     elif(self.q[0]==3):
        #         rospy.loginfo("Steve wants water.")
        #         self.q.popleft()

        #         # self.last_task_success=self.q[0]
        #         time.sleep(5)
        #     elif(self.q[0]==4):
        #         rospy.loginfo("Steve wants to talk to Harry.")
        #         self.q.popleft()

        #             # self.last_task_success=self.q[0]
        #         time.sleep(5)
        #     elif(self.q[0]==0):
                
        #         rospy.loginfo(self.q[0])
        #         rospy.loginfo("No tasks yet.")
        #         self.q.popleft()
                        # task_thread.start()
                        # task_thread = self.executor.submit(self.executeTask, 0, q[0])
                        
                # task_thread.start()
                # if self.last_task_success == self.q[0]:
                #     self.q.pop()
                #     rospy.loginfo(self.q)
                #     # rospy.sleep(5)
                #     self.last_task_success=0

       
                # self.startNavService()
                # success = self.navigate_to_location(LocationOfInterest.TABLE)
                self.bot_state.update_operation_mode(OperationModes.TELEOPERATION)
                # success = self.navigate_to_location(LocationOfInterest.HOME)

    def executeTask(self,i,j):

        # self.stow_robot_service()
        # self.startNavService()
        k=j
        if(k==1):
            rospy.loginfo("Jake requested for water.")
            self.last_task_success=k
            rospy.sleep(5)
        elif(k==2):
            rospy.loginfo("Jake wants to talk to Donna.")
            self.last_task_success=k
            rospy.sleep(5)
        elif(k==3):
            rospy.loginfo("Steve wants water.")
            self.last_task_success=k
            rospy.sleep(5)
        elif(k==4):
            rospy.loginfo("Steve wants to talk to Harry.")
            self.last_task_success=k
            rospy.sleep(5)
        elif(k==0):
            rospy.loginfo(k)
            rospy.loginfo("No tasks yet.")
        navSuccess = self.navigate_to_location(self.navigationGoal,j)
        return self.last_task_success
        
    # def jake(self):
    #     self.startNavService()
    #     navSuccess = self.navigate_to_location(self.navigationGoal)
    
    def battery_check(self, data):
        if((data.voltage)>11.8):
            self.db.child("battery_state").set(0) #updates firebase to 0 if battery above recommended threshold
        else:
            self.db.child("battery_state").set(1) #updates firebase to 1 if battery below recommended threshold


        

    def navigate_to_location(self, location : Enum, k):
        locationName = location.name
        goal = MoveBaseGoal()
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Executing task. Going to {}".format(locationName))
        # send goal
        if(k in [1,2,3,4]):
            self.db.child("task_person").set("Jake") #updates firebase to 1 if battery below recommended threshold
        elif(k in [5,6,7,8]):
            self.db.child("task_person").set("Amy") #updates firebase to 1 if battery below recommended threshold

        if(k==1):
            # goal.target_pose.pose.position.x = -23.77337646484375 #elevator
            # goal.target_pose.pose.position.y = 16.433826446533203

            self.db.child("current_action").set("Delivery") 
            self.db.child("eta").set(0) 
            goal.target_pose.pose.position.x = -3.32 # table
            goal.target_pose.pose.position.y = -9.54
            # success = self.visualServoing.main(goal.objectId)
        elif(k==2):
            # goal.target_pose.pose.position.x = -14.054288864135742 #beverly
            # goal.target_pose.pose.position.y = 10.595928192138672
            self.db.child("eta").set(0)
            self.db.child("current_action").set("Delivery") 

            goal.target_pose.pose.position.x = -2.38 # refrigerator
            goal.target_pose.pose.position.y = -7.66

        elif(k==3):
            # goal.target_pose.pose.position.x = -19.99483299255371 #aims door
            # goal.target_pose.pose.position.y = 12.28384780883789
            goal.target_pose.pose.position.x = -3.69 # center area
            self.db.child("current_action").set("Video Call") 

            goal.target_pose.pose.position.y = -3.83
            self.db.child("eta").set(0)
        elif(k==4):
            # goal.target_pose.pose.position.x = 0.8146820068359375 #lobby end
            # goal.target_pose.pose.position.y = 0.53509521484375
            goal.target_pose.pose.position.y = -4.30 # center area
            self.db.child("current_action").set("Video Call") 

            goal.target_pose.pose.position.y = -1.70
            self.db.child("eta").set(0)
        elif(k==5):
            # goal.target_pose.pose.position.x = 0.8146820068359375 #lobby end
            # goal.target_pose.pose.position.y = 0.53509521484375
            goal.target_pose.pose.position.y = -4.30
            goal.target_pose.pose.position.y = -1.70
            self.db.child("eta2").set(0)
            self.db.child("current_action").set("Delivery") 

        elif(k==6):
            # goal.target_pose.pose.position.x = 0.8146820068359375 #lobby end
            # goal.target_pose.pose.position.y = 0.53509521484375
            goal.target_pose.pose.position.y = -4.30
            goal.target_pose.pose.position.y = -1.70
            self.db.child("current_action").set("Delivery") 

            self.db.child("eta2").set(0)
        elif(k==7):
            # goal.target_pose.pose.position.x = 0.8146820068359375 #lobby end
            # goal.target_pose.pose.position.y = 0.53509521484375
            goal.target_pose.pose.position.y = -4.30
            goal.target_pose.pose.position.y = -1.70
            self.db.child("current_action").set("Video Call") 

            self.db.child("eta2").set(0)
        elif(k==8):
            # goal.target_pose.pose.position.x = 0.8146820068359375 #lobby end
            # goal.target_pose.pose.position.y = 0.53509521484375
            goal.target_pose.pose.position.y = -4.30
            goal.target_pose.pose.position.y = -1.70
            self.db.child("current_action").set("Video Call") 

            self.db.child("eta2").set(0)
        goal.target_pose.header.frame_id = "map"
        quaternion = get_quaternion(240)
        goal.target_pose.pose.orientation = quaternion
        self.navigation_client.send_goal(goal, feedback_cb = self.bot_state.navigation_feedback)
        print("Sending goal to move_base", self.goal_locations[locationName]['x'], self.goal_locations[locationName]['y'], self.goal_locations[locationName]['theta'])
        
             
        wait = self.navigation_client.wait_for_result()

        if(k in [1,2,3,4]):
            self.eta2=self.eta2-10
            self.db.child("eta2").set(self.eta2)


        elif(k in [5,6,7,8]):
            self.eta=self.eta-10
            self.db.child("eta").set(self.eta)

        if self.navigation_client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[{rospy.get_name()}]:" +"Failed to reach Table")
            # cancel navigation
            self.navigation_client.cancel_goal()
            return False
        
        success = self.visualServoing.main(0)
        if(success!=0):
            self.visualServoing.recoverFromFailure()
        self.nServoTriesAttempted += 1
        if self.nServoTriesAttempted >= self.nServoTriesAllowed:
            success = True
            self.nServoTriesAttempted=0

        self.bot_state.update_operation_mode(OperationModes.TELEOPERATION)

        rospy.loginfo(f"[{rospy.get_name()}]:" +"Reached Table")
        self.bot_state.update_emotion(Emotions.HAPPY)
        self.bot_state.update_state()
        self.bot_state.currentGlobalState = GlobalStates.REACHED_GOAL
        rospy.loginfo(f"[{rospy.get_name()}]:" +"Reached Table")
        return True

    def updateButton(self):
        while True:
            try:
                rospy.loginfo(self.q)
                # remote_callback_0 : 0
                # remote_callback_1 : 2
                # remote_callback_2 : 3
                # remote_callback_3 : 4

                # for remoteid in range(10):
                #     remoteval = self.db.child(f'remote_callback_{remoteid}').get().val()

                #     if remoteval != 0:
                #         self.q.append((remoteid, remoteval))

                button_callback_value = self.db.child("button_callback").get().val()
                button_callback_value2 = self.db.child("button_callback2").get().val()
                button_callback_value3 = self.db.child("button_callback3").get().val()
                button_callback_value4 = self.db.child("button_callback4").get().val()
                button_callback_value5 = self.db.child("button_callback5").get().val()
                button_callback_value6 = self.db.child("button_callback6").get().val()
                button_callback_value7 = self.db.child("button_callback7").get().val()
                button_callback_value8 = self.db.child("button_callback8").get().val()
                button_callback_value9 = self.db.child("button_callback9").get().val()
                button_callback_value10 = self.db.child("button_callback10").get().val()
                button_callback_value11 = self.db.child("button_callback11").get().val()
                button_callback_value12 = self.db.child("button_callback12").get().val()
                
                # p=pimu.Pimu()
                # p.pull_status()
                # if(p.status['voltage']<p.config['low_voltage_alert']):
                #     self.db.child("battery_state").set(1) #updates firebase to 1 if battery below recommended threshold
 
                if(button_callback_value==1):
                    if 1 not in self.q:
                        # self.eta=0
                        self.q.append(1)
                        if all(value != self.q[0] for value in [5,6,7,8]):
                            self.db.child("eta").set(0)
                        self.eta2 += 10
                        self.db.child("eta2").set(self.eta2)
                        
                        # self.eta=0

                        self.db.child("button_callback").set(0)

                elif(button_callback_value2==1):
                    if 2 not in self.q:
                        # self.eta=0
                        self.q.append(2)

                        if all(value != self.q[0] for value in [5,6,7,8]):
                            self.db.child("eta").set(0)
                        self.eta2 += 10
                        self.db.child("eta2").set(self.eta2)

                        # self.eta=0
                        self.db.child("button_callback2").set(0)
                elif(button_callback_value3==1):
                    if 3 not in self.q:
                        # self.eta=0
                        self.q.append(3)

                        if all(value != self.q[0] for value in [5,6,7,8]):
                            self.db.child("eta").set(0)
                        self.eta2 += 10
                        self.db.child("eta2").set(self.eta2)

                        # self.eta=0
                        self.db.child("button_callback3").set(0)
                elif(button_callback_value4==1):
                    if 4 not in self.q:
                        # self.eta=0
                        self.q.append(4)   

                        if all(value != self.q[0] for value in [5,6,7,8]):
                            self.db.child("eta").set(0)
                        self.eta2 += 10
                        # self.eta=0
                        self.db.child("eta2").set(self.eta2)
                        
                        self.db.child("button_callback4").set(0)
                elif(button_callback_value5==1):
                    if 5 not in self.q:
                        # self.eta2=0
                        self.q.append(5)   

                        if all(value != self.q[0] for value in [1, 2, 3, 4]):
                            self.db.child("eta2").set(0)
                        self.eta += 10


                        self.db.child("eta").set(self.eta)
                        self.db.child("button_callback5").set(0)
                elif(button_callback_value6==1):
                    if 6 not in self.q:
                        # self.eta2=0
                        self.q.append(6)   

                        if all(value != self.q[0] for value in [1, 2, 3, 4]):
                            self.db.child("eta2").set(0)
                        self.eta += 10
                        self.db.child("eta").set(self.eta)
                        
                        self.db.child("button_callback6").set(0)
                elif(button_callback_value7==1):
                    if 7 not in self.q:
                        # self.eta2=0
                        self.q.append(7)   

                        if all(value != self.q[0] for value in [1, 2, 3, 4]):
                            self.db.child("eta2").set(0)
                        self.eta += 10
                        self.db.child("eta").set(self.eta)

                        self.db.child("button_callback7").set(0)
                elif(button_callback_value8==1):
                    if 8 not in self.q:
                        # self.eta2=0
                        self.q.append(8)   

                        if all(value != self.q[0] for value in [1, 2, 3, 4]):
                            self.db.child("eta2").set(0)

                        self.eta += 10
                        self.db.child("eta").set(self.eta)


                        self.db.child("button_callback8").set(0)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    task_planner = TaskPlanner()
    
    # rospy.sleep(2)
    try:    
        task_planner.main()
    except KeyboardInterrupt:
        print("Shutting down")
    
    

