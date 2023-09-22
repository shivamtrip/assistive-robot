#! /usr/bin/env python3

from manip_basic_control import ManipulationMethods
from visual_servoing import AlignToObject
import rospy
import actionlib
from enum import Enum
import numpy as np
from grasp_detector.msg import GraspPoseAction, GraspPoseResult, GraspPoseGoal
from manipulation.msg import TriggerAction, TriggerFeedback, TriggerResult, TriggerGoal
import json

class States(Enum):
    IDLE = 0
    VISUAL_SERVOING = 1
    PICK = 2
    WAITING_FOR_GRASP = 3
    PLACE = 4
    COMPLETE = 5

class ManipulationFSM:

    def __init__(self):
        rospy.init_node('manipulation_fsm', anonymous=True)
        
        self.state = States.IDLE
        self.grasp = None

        self.server = actionlib.SimpleActionServer('manipulation_fsm', TriggerAction, execute_cb=self.main, auto_start=False)

        # wait for the grasp action server to boot up
        self.graspActionClient = actionlib.SimpleActionClient('grasp_detector', GraspPoseAction)
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for grasp_detector server...")
        self.graspActionClient.wait_for_server()

        self.visualServoing = AlignToObject()
        self.visualServoing.init()

        self.server.start() # start server only after everything under this is initialized
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Node Ready.")
    
    def send_feedback(self, info):
        feedback = TriggerFeedback()
        feedback.state = self.state.value
        feedback.curState = self.state.value
        feedback.info = json.dumps(info)
        self.server.publish_feedback(feedback)

    def main(self, goal : TriggerGoal):
                
        while True:
            if self.state == States.IDLE:
                self.state = States.VISUAL_SERVOING
                self.send_feedback({'msg' : "Trigger Request received. Started work."})

            elif self.state == States.VISUAL_SERVOING:
                self.visualServoing.main(goal.objectId)
                self.state = States.PICK
                self.send_feedback({'msg' : "grasing complete successfully."})

            elif self.state == States.PICK:
                self.state = States.WAITING_FOR_GRASP
                goal = GraspPoseGoal()
                self.graspActionClient.send_goal(goal)
                self.graspActionClient.wait_for_result()
                self.grasp = self.graspActionClient.get_result()
                
                if self.grasp.success == True:
                    ManipulationMethods.pick(None, self.grasp.x, abs(self.grasp.y), self.grasp.z, self.grasp.yaw)
                    self.state = States.COMPLETE

            elif self.state == States.COMPLETE:
                break


if __name__ == '__main__':
    node = ManipulationFSM()
    # goal = TriggerGoal()
    # goal.objectId = 39
    # node.main(goal)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")