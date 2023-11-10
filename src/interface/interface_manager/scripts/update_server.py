
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


        self.firebase_schema_path = os.path.expanduser("~/ws/src/interface/interface_manager/config/firebase_schema.json")



    def reset_button_state(self, button, remote, system):

        with open(self.firebase_schema_path) as f:
            self.current_cloud_status = json.load(f)

        self.current_cloud_status[system]['remote'][remote]['button'][button]['state'] = "0"

        self.update_server(self.current_cloud_status)



    def update_emotion(self, emotion: Enum):

        with open(self.firebase_schema_path) as f:
            self.current_cloud_status = json.load(f)

        self.current_cloud_status[system]['hri_params']['emotion']['name'] = emotion.name
        self.emotion = emotion
        self.update_server(self.current_cloud_status)



    def update_server(self, server_update):

        self.db.child("alfred_deployment").update(server_update)



if __name__ == "__main__":


    server_updater = ServerUpdater()

