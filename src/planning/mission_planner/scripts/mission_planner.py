#!/usr/local/lib/robot_env/bin/python3

"""
Mission Planner works as follows: 
1. Listens to new task updates from Interface Manager
2. If new task, allocate new task in task queue
3. If tasks in queue, battery health good and system is not performing another task, update next task information
4. Offload task execution to TaskExecutor and monitor progress. 
"""


from collections import deque 
import rospy
from alfred_msgs.msg import DeploymentTask, StatusMessage
from enum import Enum
from task_executor import TaskExecutor
from state_manager import BotState, TaskType, Emotions, LocationOfInterest, GlobalStates, ObjectOfInterest, VerbalResponseStates, OperationModes
import time
from sensor_msgs.msg import BatteryState


class Mission_Planner():
    
    def __init__(self):
        
        rospy.init_node("mission_planner")

        self.task_queue = deque()

        self.count = 0
        
        self.TaskExecutor = TaskExecutor()
        self.bot_state = BotState()

        self.task_listener = rospy.Subscriber('deployment_task_info', DeploymentTask, self.allocate_task)       
        self.battery_sub = rospy.Subscriber('/battery', BatteryState, self.health_manager)

        self.status_publisher = rospy.Publisher('system_status_info', StatusMessage, queue_size=10)


        self.current_task_type = ""
        self.current_task_room = ""
        self.current_task_resident = ""
        self.current_task_object = ""
        self.current_task_relative = ""


        rospy.sleep(0.5)
        self.update_mission_status(GlobalStates.IDLE)
        
        print("Mission Planner initialized..")
    

        
    def allocate_task(self, msg):
        """
        Task Allocator adds information regarding the newly added task to the task queue    
        """
        
        print("Just received a new task from the Interface Manager!")
        self.task_queue.append(msg)



    
    def health_manager(self, data):
        """
        Health Manager updates battery status (and other health parameters, if applicable) 
        """

        # print("GOT BATTERY STATE!")

        min_voltage = 11.3
        max_voltage = 13

        self.bot_state.battery_percent = int(max(min(100 * ((data.voltage - min_voltage) / (max_voltage - min_voltage)), 100), 0))

        # print("Voltage", data.voltage)
        # print("Battery", self.bot_state.battery_percent)


    def update_mission_status(self, state, mode = OperationModes.AUTONOMOUS, emotion = Emotions.HAPPY):

        print("--- Updating Mission Status ---")

        self.bot_state.emotion = emotion
        self.bot_state.global_state = state
        self.bot_state.operation_mode = mode

        self.status_update_to_interface_manager()


    
            
    def update_next_task(self):
        
        next_task = self.task_queue.popleft()
        
        self.current_task_type = next_task.task_type
        self.current_task_object = next_task.object_type
        self.current_task_resident = next_task.resident_name
        self.current_task_relative = next_task.relative_name
        self.current_task_room = next_task.room_number
        
        if self.current_task_type == TaskType.DELIVERY.name:

            self.current_task_location_1 = ObjectOfInterest.BOTTLE.name # object pickup location / Enum name (string)
            self.current_task_location_2 = self.current_task_room  # / Enum of room_number name (string)

        # elif self.current_task_type = "videocall":
        #     self.current_task_location_1 = next_task.room_number
        #     self.current_task_location_2 = next_task.room_number

        # elif self.current_task_type == "return_home":
        #     self.current_task_location_1 = # home base location
        #     self.current_task_location_2 = None

        self.status_update_to_interface_manager()


    def execute_task(self):

        print("Executing next task!", self.count)

        if self.current_task_type == TaskType.DELIVERY.name:

            
            # Navigate to Pick-up Location            
            self.update_mission_status(GlobalStates.NAVIGATION_TO_PICK)
            navSuccess = self.TaskExecutor.navigate_to_location(self.current_task_location_1)
            if not navSuccess:
                return

            print("Reached pick-up location")

            # Search for and pick-up object
            self.update_mission_status(GlobalStates.MANIPULATION_TO_PICK)
            manipulationSuccess = self.TaskExecutor.manipulate_object(ObjectOfInterest[self.current_task_object], isPick = True)
            if not manipulationSuccess:
                rospy.loginfo("Manipulation failed. Cannot find object, returning to user")


            # Navigate to delivery location
            self.update_mission_status(GlobalStates.NAVIGATION_TO_PLACE)
            navSuccess = self.TaskExecutor.navigate_to_location(self.current_task_location_2)
            if not self.update_mission_status(navSuccess):
                return 


            # # Place object at delivery location
            # manipulationSuccess = manipulate_object()
            # if not self.update_mission_status(manipulationSuccess):
            #     return 


        # elif self.current_task_type == "videocall":
        #     navigate_to_location()      # In front of door
        #     wait_for_door_open()
        #     navigate_to_location()      # At videocall spot in room 
        #     align_to_user()
        #     start_videocall()

        # elif self.current_task_type == "return_to_home_base":
        #     pass


        print("Mission Completed")



    def status_update_to_interface_manager(self):

        status_message = StatusMessage()

        status_message.last_update = time.time()
        status_message.battery_percent = self.bot_state.battery_percent
        status_message.operation_mode = self.bot_state.operation_mode.name
        status_message.global_state = self.bot_state.global_state.name
        status_message.emotion = self.bot_state.emotion.name
        status_message.music = self.bot_state.music

        status_message.task_type = self.current_task_type
        status_message.room_number = self.current_task_room
        status_message.resident_name = self.current_task_resident
        status_message.object_type = self.current_task_object
        status_message.relative_name = self.current_task_relative

        self.status_publisher.publish(status_message)

        print("Mission Planner sent status update to Interface Manager")


    def main(self):

        # if not self.battery_health_ok or not self.task_queue or system_state != STATES.IDLE:
        #     print("Cannot perform next task")
    
        rate = rospy.Rate(10)  # 10 Hz, adjust the rate as needed

        while not rospy.is_shutdown():
            if self.task_queue: # and self.system_state == States.IDLE:

                self.update_next_task()
                rospy.sleep(0.1)

                self.execute_task()
                # self.system_state = States.TASK_MODE

            rate.sleep()

            


if __name__ == "__main__":
    
    mission_planner = Mission_Planner()
    
    mission_planner.main()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
    