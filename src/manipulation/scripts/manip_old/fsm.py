#! /usr/bin/env python3

from manip_basic_control import ManipulationMethods
from visual_servoing import AlignToObject
import rospy
import actionlib
from enum import Enum
import stretch_body.robot as rb
import numpy as np
from grasp_detector.msg import GraspPoseAction, GraspPoseResult, GraspPoseGoal
from manipulation.msg import TriggerAction, TriggerFeedback, TriggerResult, TriggerGoal
import json
from control_msgs.msg import FollowJointTrajectoryAction
from plane_detector.msg import PlaneDetectAction, PlaneDetectResult, PlaneDetectGoal
from helpers import move_to_pose
from std_srvs.srv import Trigger, TriggerResponse

class States(Enum):
    IDLE = 0
    VISUAL_SERVOING = 1
    PICK = 2
    WAITING_FOR_GRASP_AND_PLANE = 3
    WAITING_FOR_GRASP = 4
    WAITING_FOR_PLANE = 5
    PLACE = 6
    COMPLETE = 7

class ManipulationFSM:

    def __init__(self, loglevel = rospy.INFO):

        rospy.init_node('manipulation_fsm', anonymous=True, log_level=loglevel)
        self.state = States.IDLE
        self.grasp = None
        self.planeOfGrasp = None
        # REFACTOR this to be a ros parameter loaded upon launch.
        self.label2name = {
            39: "bottle",
            47: "apple",
            65: "remote", 
            46: "banana"
        }

        #Initiliaze object specific params. 
        self.offset_dict = {
            'bottle': -0.015,
            'apple': -0.04,
            'remote':-0.015, 
            "banana" : -0.015
        } 

        self.isContactDict = {
            'bottle': False,
             'apple':  True,
            'remote': True,
            'banana': True
        }

        self.server = actionlib.SimpleActionServer('manipulation_fsm', TriggerAction, execute_cb=self.main, auto_start=False)

        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.graspActionClient = actionlib.SimpleActionClient('grasp_detector', GraspPoseAction)
        self.planeFitClient = actionlib.SimpleActionClient('plane_detector', PlaneDetectAction)


        # rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for trajectoryClient server...")
        # self.trajectoryClient.wait_for_server()

        # rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for grasp_detector server...")
        # self.graspActionClient.wait_for_server()

        # rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for plane_detector server...")
        # self.planeFitClient.wait_for_server()


        self.visualServoing = AlignToObject(-1)

        
        
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for stow_robot service...")
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        self.stow_robot_service.wait_for_service()


        self.manipulationMethods = ManipulationMethods()
        self.server.start() # start server only after everything under this is initialized
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Node Ready.")
    
    def send_feedback(self, info):
        feedback = TriggerFeedback()
        feedback.curState = self.state.value
        feedback.curStateInfo = json.dumps(info)
        rospy.loginfo(f"[{rospy.get_name()}]:" + json.dumps(info))
        self.server.publish_feedback(feedback)

    def requestGraspAndPlaneFit(self, getGrasp = True, getPlane = True):
        graspPoseGoal = GraspPoseGoal()
        graspPoseGoal.request = self.goal.objectId
        print("requesting grasp for object", self.goal.objectId)
        self.send_feedback({"msg": "requesting grasp for object", "data": {"objectId": self.goal.objectId}})
        graspSuccess = not getGrasp
        planeSuccess = not getPlane

        self.planeOfGrasp = None
        self.grasp = None

        planeFitGoal = PlaneDetectGoal()
        while not (graspSuccess and planeSuccess):
            if self.grasp is not None and self.grasp.success == True:
                graspSuccess = True
                self.state = States.WAITING_FOR_PLANE
            elif getGrasp:
                self.graspActionClient.send_goal(graspPoseGoal)
                self.graspActionClient.wait_for_result()
                self.grasp = self.graspActionClient.get_result()
                rospy.loginfo(f"Received grasp = {self.grasp}", )

            if self.planeOfGrasp is not None and self.planeOfGrasp.success == True:
                planeSuccess = True
                self.state = States.WAITING_FOR_GRASP
            elif getPlane:
                self.planeFitClient.send_goal(planeFitGoal)
                self.planeFitClient.wait_for_result()
                self.planeOfGrasp = self.planeFitClient.get_result()
                rospy.loginfo(f"Received plane = {self.planeOfGrasp}", )

        if getGrasp and getPlane:
            self.heightOfObject = abs(self.grasp.z - self.planeOfGrasp.z)
        
        return self.grasp, self.planeOfGrasp

    def estimatePlacingLocation(self, planeOfGrasp, heightOfObject):
        x, y, z, a, b, c, d = planeOfGrasp.x, planeOfGrasp.y, planeOfGrasp.z, planeOfGrasp.a, planeOfGrasp.b, planeOfGrasp.c, planeOfGrasp.d
        placingLocation = np.array([
            x, 
            abs(abs(y) - self.manipulationMethods.getEndEffectorPose()[0]), 
            # 1.0,
            # 0.85
            z
        ])
        placingLocation[2] += heightOfObject
        
        return placingLocation

    def reset(self):
        self.state = States.IDLE
        self.grasp = None
        self.planeOfGrasp = None

    def main(self, goal : TriggerGoal):
        self.goal = goal
        self.state = States.IDLE
        print("Requested goal", goal)
        objectManipulationState = States.PICK
        self.visualServoing.objectId = goal.objectId
        rospy.loginfo("Stowing robot prior to manipulation!")
        self.stow_robot_service()
        if goal.isPick:
            rospy.loginfo("Received pick request.")
            objectManipulationState = States.PICK
        else:
            rospy.loginfo("Received place request.")
            objectManipulationState = States.PLACE


        nServoTriesAttempted = 0
        nServoTriesAllowed = 2

        nPickTriesAttempted = 0
        nPickTriesAllowed = 2

        try: 
            while True:
                if self.state == States.IDLE:
                    self.state = States.VISUAL_SERVOING
                    if objectManipulationState == States.PLACE:
                        self.send_feedback({'msg' : "Trigger Request received. Placing"})
                        self.state = States.PLACE
                    else:
                        self.send_feedback({'msg' : "Trigger Request received. Starting to find the object"})
                elif self.state == States.VISUAL_SERVOING:
                    success = self.visualServoing.main(goal.objectId)
                    if success:
                        self.send_feedback({'msg' : "Servoing succeeded! Starting manipulation."})
                        self.state = objectManipulationState
                    else:
                        #success = self.visualServoing.main(goal.objectId)
                        if nServoTriesAttempted >= nServoTriesAllowed:
                            self.send_feedback({'msg' : "Servoing failed. Aborting."})
                            self.reset()
                            return TriggerResult(success = False)
                        
                        self.send_feedback({'msg' : "Servoing failed. Attempting to recover from failure."  + str(nServoTriesAttempted) + " of " + str(nServoTriesAllowed) + " allowed."})
                        nServoTriesAttempted += 1
                        self.visualServoing.recoverFromFailure()

                elif self.state == States.PICK:
                    self.state = States.WAITING_FOR_GRASP_AND_PLANE
                    self.send_feedback({'msg' : "moving to pregrasp pose"})
                    self.manipulationMethods.move_to_pregrasp(self.trajectoryClient)
                    # self.visualServoing.alignObjectHorizontal(offset = self.manipulationMethods.getEndEffectorPose()[1])
                    # self.requestGraspAndPlaneFit(getGrasp = True, getPlane = True)   
                    self.grasp, self.planeOfGrasp = self.requestGraspAndPlaneFit(getGrasp = True, getPlane = False)   
                    distToExtend = abs(abs(self.grasp.y) - self.manipulationMethods.getEndEffectorPose()[0]) + self.offset_dict[self.label2name[goal.objectId]]
                    self.manipulationMethods.pick(
                        self.trajectoryClient, 
                        self.grasp.x, 
                        distToExtend, 
                        self.grasp.z, 
                        self.grasp.yaw,
                        moveUntilContact = self.isContactDict[self.label2name[goal.objectId]]
                    ) 
                    success = self.manipulationMethods.checkIfGraspSucceeded()
                    if success:
                        self.send_feedback({'msg' : "Pick succeeded! Starting to place."})
                        self.state = States.COMPLETE
                    else:
                        self.send_feedback({'msg' : "Pick failed. Reattempting."})
                        if nPickTriesAttempted >= nPickTriesAllowed:
                            self.send_feedback({'msg' : "Pick failed. Cannot grasp successfully. Aborting."})
                            self.reset()
                            return TriggerResult(success = False)
                        
                        self.send_feedback({'msg' : "Picking failed. Reattempting pick, try number " + str(nPickTriesAttempted) + " of " + str(nPickTriesAllowed) + " allowed."})
                        nPickTriesAttempted += 1

                elif self.state == States.PLACE:
                    self.state = States.WAITING_FOR_GRASP_AND_PLANE
                    # self.grasp, self.planeOfGrasp = self.requestGraspAndPlaneFit(getGrasp = False, getPlane = True)
                    self.state = States.PLACE
                    heightOfObject = goal.heightOfObject
                    move_to_pose(self.trajectoryClient, {
                        'head_pan;to' : -np.pi/2,
                    })
                    rospy.sleep(0.5)
                    move_to_pose(
                        self.trajectoryClient,
                        {
                            'base_rotate;by': np.pi/2,
                        }
                    )
                    # placingLocation = self.estimatePlacingLocation(self.planeOfGrasp, heightOfObject)
                    placingLocation = np.array([0.0, 1.0, 1.1])
                    
                    self.manipulationMethods.place(self.trajectoryClient, placingLocation[0], placingLocation[1], placingLocation[2], 0)
                    self.stow_robot_service()
                    self.state = States.COMPLETE

                elif self.state == States.COMPLETE:
                    # self.send_feedback({'msg' : "Work complete successfully."})
                    rospy.loginfo(f"{rospy.get_name()} : Work complete successfully.")
                    move_to_pose(self.trajectoryClient, {
                        "head_pan;to" : 0,
                    })
                    break

        except KeyboardInterrupt:
            print("User Exited!!")

        # self.heightOfObject = abs(self.grasp.z - self.planeOfGrasp.z) 
        self.heightOfObject = 0
        self.reset()
        self.server.set_succeeded(result = TriggerResult(success = True, heightOfObject = self.heightOfObject))


   


if __name__ == '__main__':

    node = ManipulationFSM()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

