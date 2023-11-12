#!/usr/local/lib/robot_env/bin/python3

"""
This script reads the latest state of the database from config/firebase_schema.json and informs mission planner whenever a button is pressed by a resident
"""


import rospy
import os
import json
import pyrebase
import time
from collections import defaultdict
from alfred_msgs.msg import DeploymentTask, StatusMessage
from update_server import ServerUpdater


class InterfaceManager:
    
    def __init__(self):
        
        rospy.init_node("interface_manager")

        self.system_id = os.environ["HELLO_FLEET_ID"]

        self.server_updater = ServerUpdater()      
         
        self.firebase_schema_path = os.path.expanduser("~/ws/src/interface/interface_manager/config/firebase_schema.json")
        
        self.task_publisher = rospy.Publisher('deployment_task_info', DeploymentTask, queue_size=10)

        self.status_subscriber = rospy.Subscriber('system_status_info', StatusMessage, self.server_updater.update_system_status)
         
        with open(self.firebase_schema_path) as f:
            self.initial_cloud_status = json.load(f)
            
        self.system_list = []
        self.remote_list = []
        self.button_list = []

        self.initialize_lists()
            
        self.update_delay = 2       # seconds
          
        print("Interface Manager initialized..")
    

          
    def initialize_lists(self):
        
        for system in self.initial_cloud_status.keys():
            self.system_list.append(system)
            
            for remote in self.initial_cloud_status[system]['remote'].keys():
                self.remote_list.append(remote)
                
        for button in self.initial_cloud_status[self.system_list[0]]['remote'][self.remote_list[0]]['button'].keys():
            self.button_list.append(button)
                    


    def process_updates(self):       
        
        while not rospy.is_shutdown():
            
            with open(self.firebase_schema_path) as f:
                self.current_cloud_status = json.load(f)

            for system in self.system_list:

                if self.current_cloud_status[system]['id'] == self.system_id:

                    for remote in self.remote_list:
                        for button in self.button_list:
                            
                            if self.current_cloud_status[system]['remote'][remote]['button'][button]['state'] == "1":
                                
                                print(f"{button} -- {remote} -- {system} is ON")
                                
                                self.inform_mission_planner(button, remote, system)
                
                self.server_updater.system_number = system

                break
            rospy.sleep(self.update_delay)
                            
                            
    def inform_mission_planner(self, button, remote, system):
        
        task_message = DeploymentTask()
        
        task_message.task_type = self.current_cloud_status[system]['remote'][remote]['button'][button]['task_type']
        task_message.room_number = self.current_cloud_status[system]['remote'][remote]['room_number']
        task_message.resident_name = self.current_cloud_status[system]['remote'][remote]['resident_name']
        task_message.object_type = self.current_cloud_status[system]['remote'][remote]['button'][button]['object_type']
        task_message.relative_name = self.current_cloud_status[system]['remote'][remote]['button'][button]['relative_name']


        print("MESSAGE", task_message)


        self.task_publisher.publish(task_message)

        self.server_updater.reset_button_state(button, remote, system)

        




if __name__ == "__main__":
    
    server = InterfaceManager()
    
    server.process_updates()
    

    rospy.spin()

    print("Shutting down")