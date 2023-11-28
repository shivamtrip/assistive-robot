
import pyrebase
import json
import time
import numpy as np
import os 
import rospy
import json
import os
class FirebaseNode:
    def __init__(self):
        firebase_secrets_path = os.path.expanduser("~/.alfred-auxilio-firebase-adminsdk.json")
        if not os.path.isfile(firebase_secrets_path):
            raise FileNotFoundError("Firebase secrets file not found")
        
        with open(firebase_secrets_path, 'r') as f:
            config = json.load(f)

        self.firebase = pyrebase.initialize_app(config)
        self.db = self.firebase.database()
        
        self.root = "fvd_encore"
        with open('/home/hello-robot/alfred-autonomy/src/interface/alfred_hri/config/firebase_schema.json', 'r') as f:
            schema = json.load(f)

        self.db.update({self.root: schema})

    def update_node(self, dict):
        for key in dict.keys():
            self.db.update({self.root + "/" + key : dict[key]})
            
    def update_param(self, key, value):
        
        self.db.update({self.root + "/" + key : value})

if __name__ == "__main__":
    with open(os.path.expanduser("~/alfred-autonomy/src/planning/task_planner/config/firebase_schema.json")) as f:
        state_dict = json.load(f)
    def noo(s):
        pass
    node = FirebaseNode(state_dict, noo, noo)

    
    


    