
from enum import Enum
import json
from move_base_msgs.msg import MoveBaseActionFeedback, MoveBaseFeedback
import time
import rospy
from firebase_node import FirebaseNode
import os
from sensor_msgs.msg import JointState


class BotStateManager():
    def __init__(self, updateCallback):

        self.position = None
        self.orientation = None
        self.currentSemanticLocation = LocationOfInterest.HOME # GEOFENCE THIS USING DEFINED BOXES TODO(@shivamtrip)
        self.navigationOrigin = LocationOfInterest.HOME
        self.navigationGoal = LocationOfInterest.HOME
        self.objectOfInterest = ObjectOfInterest.NONE
        self.currentGlobalState = GlobalStates.WAITING_FOR_COMMAND
        self.operationMode = OperationModes.AUTONOMOUS
        self.emotion = Emotions.NEUTRAL

        with open(os.path.expanduser("~/alfred-autonomy/src/planning/task_planner/config/firebase_schema.json")) as f:
            self.state_dict = json.load(f)
        
    

        self.firebaseNode = FirebaseNode(
            self.state_dict, 
            self.get_operation_state_from_firebase, 
            self.get_teleop_commands_from_firebase
        )
        self.jointStateSubscriber = rospy.Subscriber("/alfred/joint_states", JointState, self.joint_state_callback) 
        self.last_update_time = time.time()
        self.updateCallback = updateCallback


    def isAutonomous(self):
        self.pollOperationState()
        return self.operationMode == OperationModes.AUTONOMOUS

    def update_hri(self, speech: str, emotion: Enum):
        self.state_dict['hri_params']['speech_in']['text'] = speech
        self.state_dict['hri_params']['emotion']['name'] = emotion.name
        self.emotion = emotion
        self.firebaseNode.update_node(self.state_dict)
    
    def update_operation_mode(self, mode: Enum):
        rospy.loginfo(f"Updating operation mode to {mode.name}")
        self.state_dict['operation_mode'] = str(mode.name)
        self.operationMode = mode
        self.firebaseNode.update_operation_mode({
            'operation_mode': mode.name
        })

    def update_speech(self, speech: str):
        self.state_dict['hri_params']['speech_in']['text'] = speech
        self.firebaseNode.update_node(self.state_dict)

    def update_emotion(self, emotion: Enum):
        self.state_dict['hri_params']['emotion']['name'] = emotion.name
        self.emotion = emotion
        self.firebaseNode.update_node(self.state_dict)
    def pollOperationState(self):
        mode = self.firebaseNode.db.get().val()['operation_mode']
        if mode == "AUTONOMOUS":
            self.operationMode = OperationModes.AUTONOMOUS
        else:
            self.operationMode = OperationModes.TELEOPERATION

    def get_operation_state_from_firebase(self, msg):
        self.state_dict['operation_mode'] = msg['data']
        print("Received operation mode from firebase: {}".format(self.state_dict['operation_mode']))
        if msg['data'] == "AUTONOMOUS":
            self.operationMode = OperationModes.AUTONOMOUS
        else:
            self.operationMode = OperationModes.TELEOPERATION
        self.updateCallback()

    def get_teleop_commands_from_firebase(self, msg, callback = True):
        self.state_dict['teleop_commands'] = self.firebaseNode.db.get().val()['teleop_commands']
        commands = self.firebaseNode.db.get().val()['teleop_commands']
        mode = self.firebaseNode.db.get().val()['operation_mode']
        if mode == "AUTONOMOUS":
            self.operationMode = OperationModes.AUTONOMOUS
        else:
            self.operationMode = OperationModes.TELEOPERATION
        if callback:
            self.updateCallback()
        return commands
        
        

    def joint_state_callback(self, msg : JointState):
        
        for name in msg.name:
            self.state_dict['bot_joint_params'][name] = {
                "position": msg.position[msg.name.index(name)],
                "velocity": msg.velocity[msg.name.index(name)],
                "effort": msg.effort[msg.name.index(name)]
            }
        self.firebaseNode.update_node(self.state_dict)

    
    
    def isExtendedCommunicationLossTeleop(self):
        return time.time() - (self.state_dict['teleop_commands']['last_command_stamp']) > 10

    def update_state(self):
        self.state_dict["global_state"] = self.currentGlobalState.name
        self.state_dict['current_task']["origin"] = self.navigationOrigin.name
        self.state_dict['current_task']["goal"] = self.navigationGoal.name
        self.state_dict['current_task']["object_of_interest"] = self.objectOfInterest.name

        # emotion params

    def navigation_feedback(self, msg : MoveBaseFeedback):
        self.position = msg.base_position.pose.position
        self.orientation = msg.base_position.pose.orientation
        
        # if time.time() - self.last_update_time > 1:
        #     self.last_update_time = time.time()
        rospy.logdebug(f"[{rospy.get_name()}]:" +"Current position: {}".format(self.position))
        rospy.logdebug(f"[{rospy.get_name()}]:" +"Current orientation: {}".format(self.orientation))

    def manipulation_feedback(self, feedback):
        pass


class Emotions(Enum):
    HAPPY = 0
    SAD = 1
    ANGRY = 2
    DROWSY = 3
    BORED = 4
    NEUTRAL = 5
    ATTENTION = 6

class OperationModes(Enum):
    TELEOPERATION = 0
    AUTONOMOUS = 1

class GlobalStates(Enum):
    IDLE = 10
    BOOTING = 0
    WAITING_FOR_COMMAND = 1
    RECEIVED_COMMAND = 2
    NAVIGATING = 3
    REACHED_GOAL = 4
    RECOVERY = 9
    MANIPULATION = 6
    COMPLETED_TASK = 7
    CALL = 8
    

class ObjectOfInterest(Enum): 
    # FILL WITH CLASS LABELS FOR THE OBJECT DETECTION PIPELINE
    NONE = -1
    USER = 0
    BOTTLE = 39
    BOX = 2
    GLASS = 3
    TABLE = 60
    REMOTE = 65
    APPLE = 47
    BANANA = 46
    

class LocationOfInterest(Enum):
    # FILL WITH LOCATION LABELS FROM GROUNDING PIPELINE
    HOME = -1
    LIVING_ROOM = 0
    KITCHEN = 1
    TABLE = 2
    NET = 3

class VerbalResponseStates(Enum):
    # FILL WITH VERBAL RESPONSES
    NONE = -1
    UHUH = 0
    OK = 1
    ON_IT = 2
    SORRY = 3
    THANKS = 4
    YES = 5
    HERE_YOU_GO = 6
