#!/usr/local/lib/robot_env/bin/python3

"""
This script reads from the firebase server and updates the latest state of the database in config/firebase_schema.json
"""


import rospy
from firebase_node import FirebaseNode
import os
import json
import pyrebase
import time

class CloudListener:
    
    def __init__(self):
        
        rospy.init_node("cloud_listener")
        
        firebase_secrets_path = os.path.expanduser("~/firebasesecrets.json")
        if not os.path.isfile(firebase_secrets_path):
            raise FileNotFoundError("Firebase secrets file not found")
        
        with open(firebase_secrets_path, 'r') as f:
            config = json.load(f)
            
        
        self.firebase = pyrebase.initialize_app(config)
        self.db = self.firebase.database()
            
        
        self.firebase_schema_path = os.path.expanduser("~/ws/src/interface/interface_manager/config/firebase_schema.json")
            
        if not os.path.exists(self.firebase_schema_path):
                open(self.firebase_schema_path, 'w')
        
    
        self.update_delay = 2       # seconds
        
            
        
    def read_server(self):
                
        while not rospy.is_shutdown():
        
            deployment_server_json = self.db.child("alfred_deployment").get().val()
            
            with open(self.firebase_schema_path, "w") as f:
                json.dump(deployment_server_json, f, indent = 4)
            
            print("Updated latest server state")
            rospy.sleep(self.update_delay)
        
        
        
        
if __name__ == "__main__":
    
    server = CloudListener()
    
    server.read_server()

    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")