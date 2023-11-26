#!/usr/bin/env python3

# from control_msgs.msg import FollowJointTrajectoryGoal
import rospy
from std_srvs.srv import Trigger, TriggerResponse
# from std_msgs.msg import String, Header
import time
import numpy as np
from sensor_msgs.msg import JointState, CameraInfo, Image
from enum import Enum
from scipy.spatial.transform import Rotation as R
from control_msgs.msg import FollowJointTrajectoryAction
import actionlib
from visual_servoing import AlignToObject
from manipulation.msg import TriggerAction, TriggerFeedback, TriggerResult, TriggerGoal
from human_detection import HumanDetector
from std_msgs.msg import Bool


class State(Enum):
    SEARCH = 1
    
    MOVE_TO_OBJECT = 3
    COMPLETE = 7

    # substates
    SCANNING = 11      
    ALIGN_TO_OBJECT = 12          

    FAILED = -1

class Error(Enum):
    CLEAN = 0
    SCAN_FAILED = 1
    


class TelepresenceManager():

    def __init__(self):

        self.prev_state = None
        self.current_state = None

        self.yolo_status_control_pub = rospy.Publisher("yolo_status_control", Bool, queue_size=10)

        startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        startManipService()

        self.trajectoryClient = actionlib.SimpleActionClient('/alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        print("Waiting for trajectory server")
        self.trajectoryClient.wait_for_server()

        self.tele_man_server = actionlib.SimpleActionServer('telepresence_manager', TriggerAction, execute_cb=self.main, auto_start=False)

        self.human_detector = HumanDetector()
        self.visual_servoing = AlignToObject(self.trajectoryClient, self.human_detector)

        rospy.loginfo("waiting for stow robot service")
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        self.stow_robot_service.wait_for_service()
        self.stow_robot_service()

        self.tele_man_server.start()

        self.tele_man_result = TriggerResult()
   
        rospy.loginfo("Telepresence Manager is ready!\n")



    def main(self, objectId):
        
        print("Telepresence Manager received goal from Mission Planner")


        self.yolo_status_control_pub.publish(True)      
    

        self.human_detector.objectId = objectId

        self.current_state = State.SEARCH
        self.num_retries = 0

        while self.current_state != State.COMPLETE and self.num_retries < 5:

            self.prev_state = self.current_state

            if self.current_state == State.SEARCH:
                success = self.visual_servoing.findAndOrientToObject()                
                self.current_state = State.MOVE_TO_OBJECT if success else State.FAILED

            elif self.current_state == State.MOVE_TO_OBJECT:
                success = self.visual_servoing.moveTowardsObject()
                self.current_state = State.COMPLETE if success else State.FAILED

            rospy.loginfo(f"Completed {self.prev_state.name}. Will now do {self.current_state.name}")
            rospy.sleep(2)


            if self.current_state == State.FAILED:
                # Stop ArUco detector
                print("Stopping Alignment to User")
                print(f"Aligning to user failed during the {self.prev_state.name} state.")
                self.tele_man_result.success = False
                self.tele_man_server.set_succeeded(self.tele_man_result)

                if self.num_retries > 5:
                    self.human_detector.objectId = None
                    self.yolo_status_control_pub.publish(False)
                    return False

                self.num_retries += 1
                self.current_state = State.SEARCH 

        # Stop ArUco detector
        print("Stopping Alignment to User")
        self.yolo_status_control_pub.publish(False)
        self.human_detector.objectId = None
        self.tele_man_result.success = True
        self.tele_man_server.set_succeeded(self.tele_man_result)
        print("Successfully completed manipulation!")
        return True


if __name__ == "__main__":
    switch_to_manipulation = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    switch_to_manipulation.wait_for_service()
    switch_to_manipulation()
    rospy.init_node("align_to_object", anonymous=True)
    node = TelepresenceManager()
    node.main(0)

    # rospy.init_node('telepresence_manager')
    # telepresence_manager = TelepresenceManager()
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass