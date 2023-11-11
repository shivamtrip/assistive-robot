
"""
This script is responsible for updating any state changes to firebase
"""



import rospy
import os
import json
import pyrebase
import time



class ServerUpdater:

    def __init__(self):

        firebase_secrets_path = os.path.expanduser("~/firebasesecrets.json")
        if not os.path.isfile(firebase_secrets_path):
            raise FileNotFoundError("Firebase secrets file not found")
        
        with open(firebase_secrets_path, 'r') as f:
            config = json.load(f)

        
        self.firebase = pyrebase.initialize_app(config)
        self.db = self.firebase.database()

        self.system_number = None

        self.firebase_schema_path = os.path.expanduser("~/ws/src/interface/interface_manager/config/firebase_schema.json")



    def reset_button_state(self, button, remote, system):

        with open(self.firebase_schema_path) as f:
            self.current_cloud_status = json.load(f)

        self.current_cloud_status[system]['remote'][remote]['button'][button]['state'] = "0"

        self.update_server(self.current_cloud_status)



    def update_system_status(self, status_message):

        with open(self.firebase_schema_path) as f:
            self.current_cloud_status = json.load(f)

        print("Interface Manager now updating System Status")

        self.current_cloud_status[self.system_number]['status']['battery_percent'] = status_message.battery_percent
        self.current_cloud_status[self.system_number]['status']['last_update'] = status_message.last_update
        self.current_cloud_status[self.system_number]['status']['operation_mode'] = status_message.operation_mode
        self.current_cloud_status[self.system_number]['status']['global_state'] = status_message.global_state

        self.current_cloud_status[self.system_number]['status']['current_task']['object_type'] = status_message.object_type
        self.current_cloud_status[self.system_number]['status']['current_task']['relative_name'] = status_message.relative_name
        self.current_cloud_status[self.system_number]['status']['current_task']['resident_name'] = status_message.resident_name
        self.current_cloud_status[self.system_number]['status']['current_task']['room_number'] = status_message.room_number
        self.current_cloud_status[self.system_number]['status']['current_task']['task_type'] = status_message.task_type

        self.current_cloud_status[self.system_number]['status']['hri_params']['emotion'] = status_message.emotion
        self.current_cloud_status[self.system_number]['status']['hri_params']['music'] = status_message.music

        self.update_server(self.current_cloud_status)




    def update_server(self, server_update):

        self.db.child("alfred_deployment").update(server_update)



if __name__ == "__main__":


    server_updater = ServerUpdater()

