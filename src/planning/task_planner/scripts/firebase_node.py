
import pyrebase
import json
import time
import numpy as np
import os 
import rospy
import json
import os
class FirebaseNode:
    def __init__(self, schema, operation_mode_callback, teleop_mode_callback):
        firebase_secrets_path = os.path.expanduser("~/.alfred-auxilio-firebase-adminsdk.json")
        if not os.path.isfile(firebase_secrets_path):
            raise FileNotFoundError("Firebase secrets file not found")
        
        with open(firebase_secrets_path, 'r') as f:
            config = json.load(f)

        self.firebase = pyrebase.initialize_app(config)
        self.db = self.firebase.database()
        self.root = "alfred_fvd"
        
        self.db.update({self.root: schema})

        self.last_update = time.time()
        rospy.sleep(2)
        self.operation_stream = self.db.child(self.root + "/operation_mode").stream(operation_mode_callback)
        self.teleoperation_stream = self.db.child(self.root + "/teleop_commands").stream(teleop_mode_callback)
    # def update_operation_mode(self, mode):
    #     self.db.update(mode)


    def update_node(self, dict):
        dict['last_update'] = (time.time())
        
        if time.time() - self.last_update > 2:
            # keys_to_del = [
            #     'operation_mode',
            #     'teleop_commands'
            # ]
            # for key in keys_to_del:
            #     if key in dict.keys():
            #         del dict[key]

            for key in dict.keys():
                if key != 'operation_mode' and key != 'teleop_commands':
                    self.db.update({self.root + "/" + key : dict[key]})
            # self.db.update({self.root : dict})
            self.last_update = time.time()

if __name__ == "__main__":
    with open(os.path.expanduser("~/alfred-autonomy/src/planning/task_planner/config/firebase_schema.json")) as f:
        state_dict = json.load(f)
    def noo(s):
        pass
    node = FirebaseNode(state_dict, noo, noo)

    
    


    