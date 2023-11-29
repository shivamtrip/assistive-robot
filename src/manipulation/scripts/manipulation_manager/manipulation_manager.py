#!/usr/bin/env python3

from enum import Enum
import rospy
from manipulation.msg import TriggerAction, TriggerFeedback, TriggerResult, TriggerGoal
from std_srvs.srv import Trigger
import actionlib
from aruco_detection import ArUcoDetector
from visual_servoing import AlignToObject
from manipulation_methods import ManipulationMethods
from control_msgs.msg import FollowJointTrajectoryAction
import numpy as np
# from manipulation.srv import ArUcoStatus, ArUcoStatusResponse

class State(Enum):
    SEARCH = 1
    MOVE_TO_OBJECT = 3
    ROTATE_90 = 4
    ALIGN_HORIZONTAL = 5
    COMPLETE = 6
    PICK = 8
    NAVIGATION = 9
    PLACE = 10
    SCANNING = 11      
    ALIGN_TO_OBJECT = 12          
    FAILED = -1




class ManipulationManager:

    def __init__(self):

        self.prev_state = None
        self.current_state = None
        self.manipulation_state = None

        startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        startManipService()

        self.trajectoryClient = actionlib.SimpleActionClient('/alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        print("Waiting for trajectory server")
        self.trajectoryClient.wait_for_server()

        self.manip_man_server = actionlib.SimpleActionServer('manipulation_manager', TriggerAction, execute_cb=self.main, auto_start=False)

        self.aruco_detector = ArUcoDetector()
        self.visual_servoing = AlignToObject(self.trajectoryClient, self.aruco_detector)
        self.manipulation_methods = ManipulationMethods()

        # self.aruco_status_service = rospy.ServiceProxy('/start_aruco_detection', ArUcoStatus)
        # self.aruco_status_service.wait_for_service()
        # self.aruco_status_message = ArUcoStatus()

        rospy.loginfo("waiting for stow robot service")
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        self.stow_robot_service.wait_for_service()
        self.stow_robot_service()

        self.manip_man_server.start()

        self.manip_man_result = TriggerResult()
        
        rospy.loginfo("Manipulation Manager is ready\n")


    def execute_manipulation(self):

        aruco_result = self.aruco_detector.getArucoLocation()

        if aruco_result:
            
            [x, y, z] = aruco_result

            if self.manipulation_state == State.PICK:
                ee_x, ee_y, ee_z = self.manipulation_methods.getEndEffectorPose()
                distToExtend = abs(y - ee_y)
                self.manipulation_methods.pick(self.trajectoryClient, x, distToExtend, z, 0, moveUntilContact = False) 
            

            elif self.manipulation_state == State.PLACE:
                ee_x, ee_y, ee_z = self.manipulation_methods.getEndEffectorPose()
                distToExtend = abs(y - ee_y) 
                self.manipulation_methods.place(self.trajectoryClient, x, distToExtend, z, 0, place = True) 

            self.stow_robot_service()
            return True

        else:
            return False
        

    def main(self,  goal: TriggerGoal):
        
        print("Manipulation Manager received goal from Mission Planner")

        # Start ArUco detector
        print("Starting Aruco Detections")
        self.aruco_detector.detect_aruco = True
        
        # self.aruco_detector.aruco_id = goal.objectId
        # self.aruco_detector.first_print = False
        # self.aruco_status_message.status = True
        # self.aruco_status_service(self.aruco_status_message)

        self.current_state = State.SEARCH
        self.manipulation_state = State.PICK if goal.isPick else State.PLACE

        self.num_retries = 0

        while self.current_state != State.COMPLETE:

            self.prev_state = self.current_state

            if self.current_state == State.SEARCH:
                success = self.visual_servoing.findAndOrientToObject()                
                self.current_state = State.MOVE_TO_OBJECT if success else State.FAILED

            elif self.current_state == State.MOVE_TO_OBJECT:
                success = self.visual_servoing.moveTowardsObject()
                self.current_state = State.ROTATE_90 if success else State.FAILED

            elif self.current_state == State.ROTATE_90:
                success = self.visual_servoing.alignObjectForManipulation()
                self.current_state = State.ALIGN_HORIZONTAL if success else State.FAILED

            elif self.current_state == State.ALIGN_HORIZONTAL:
                success = self.visual_servoing.alignObjectHorizontal(self.trajectoryClient) # Fixing the displacement in x was 0.07, now is 0.055
                self.current_state = self.manipulation_state if success else State.FAILED

            elif self.current_state == State.PICK:
                success = self.execute_manipulation()
                self.current_state = State.COMPLETE
                
            elif self.current_state == State.PLACE:
                success = self.execute_manipulation()
                self.current_state = State.COMPLETE


            rospy.loginfo(f"Completed {self.prev_state.name}. Will now do {self.current_state.name}")
            rospy.sleep(2)
      

            if self.current_state == State.FAILED:
                # Stop ArUco detector
                print("Stopping Aruco Detections")
                
                # self.aruco_detector.first_print = False
                # self.aruco_status_message.status = False
                # self.aruco_status_service(self.aruco_status_message)

                if self.num_retries > 5:
                    self.aruco_detector.detect_aruco = False
                    print(f"Manipulation failed during the {self.prev_state.name} state.")
                    self.manip_man_result.success = False
                    self.manip_man_server.set_succeeded(self.manip_man_result)
                    return False

                self.num_retries += 1
                self.current_state = State.SEARCH 

        # Stop ArUco detector
        print("Stopping Aruco Detections")
        self.aruco_detector.detect_aruco = False
        # self.aruco_detector.first_print = False
        # self.aruco_status_message.status = False
        # self.aruco_status_service(self.aruco_status_message)
        self.manip_man_result.success = True
        self.manip_man_server.set_succeeded(self.manip_man_result)
        print("Successfully completed manipulation!")
        return True
        
        


if __name__ == '__main__':
    rospy.init_node('manipulation_manager')
    manipulation_manager = ManipulationManager()

    # goal= TriggerGoal()
    # goal.isPick= True
    # # goal.isPick = False
    # manipulation_manager.main(goal)
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass