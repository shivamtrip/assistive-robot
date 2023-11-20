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
from control_msgs.msg import FollowJointTrajectoryAction
from plane_detector.msg import PlaneDetectAction, PlaneDetectResult, PlaneDetectGoal
from helpers import move_to_pose
from std_srvs.srv import Trigger, TriggerResponse
from scene_parser import SceneParser

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

        
        self.class_list = rospy.get_param('/object_detection/class_list')
        self.label2name = {i : self.class_list[i] for i in range(len(self.class_list))}

        self.offset_dict = rospy.get_param('/manipulation/offsets')
        self.isContactDict = rospy.get_param('/manipulation/contact')
        
        self.n_max_servo_attempts = rospy.get_param('/manipulation/n_max_servo_attempts')   
        self.n_max_pick_attempts = rospy.get_param('/manipulation/n_max_pick_attempts')

        self.server = actionlib.SimpleActionServer('manipulation_fsm', TriggerAction, execute_cb=self.main, auto_start=False)

        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for trajectoryClient server...")
        self.trajectoryClient.wait_for_server()


        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for stow_robot service...")
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        self.stow_robot_service.wait_for_service()
        
        self.scene_parser = SceneParser()

        self.visualServoing = AlignToObject(self.scene_parser)
        self.manipulationMethods = ManipulationMethods()
        
        
        self.server.start() # start server only after everything under this is initialized
        rospy.loginfo(f"[{rospy.get_name()}]:" + "Node Ready.")
    
    def send_feedback(self, info):
        feedback = TriggerFeedback()
        feedback.curState = self.state.value
        feedback.curStateInfo = json.dumps(info)
        rospy.loginfo(f"[{rospy.get_name()}]:" + json.dumps(info))
        self.server.publish_feedback(feedback)

    def reset(self):
        self.state = States.IDLE
        self.grasp = None
        self.planeOfGrasp = None

    def main(self, goal : TriggerGoal):
        self.goal = goal
        self.state = States.IDLE
        objectManipulationState = States.PICK

        self.scene_parser.set_object_id(goal.objectId)
        
        rospy.loginfo(f"{rospy.get_name()} : Stowing robot.")
        # self.stow_robot_service()
        
        if goal.isPick:
            rospy.loginfo("Received pick request.")
            objectManipulationState = States.PICK
        else:
            rospy.loginfo("Received place request.")
            objectManipulationState = States.PLACE

        nServoTriesAttempted = 0
        nServoTriesAllowed = self.n_max_servo_attempts

        nPickTriesAttempted = 0
        nPickTriesAllowed = self.n_max_pick_attempts
        # move_to_pose(
        #     self.trajectoryClient,
        #     {
        #         'head_pan;to' : -np.pi/2,
        #         'head_tilt;to' : - 30 * np.pi/180,
        #         # 'base_rotate;by': np.pi/2,
        #     }
        # )
        # while not rospy.is_shutdown():
        #     self.scene_parser.set_point_cloud(publish = True) #converts depth image into point cloud
        #     grasp = self.scene_parser.get_grasp(publish = True)
        #     print(grasp[1] * 180 / np.pi)
        #     move_to_pose(
        #         self.trajectoryClient,
        #         {
        #             'wrist_yaw;to': grasp[1],
        #         }
        #     )
        #     rospy.sleep(2)
        # exit()

        # move_to_pose(trajectoryClient = self.trajectoryClient, pose = {
        #     "head_tilt;to" : -10*np.pi/180,   
        # }
        # )
        # self.scene_parser.set_point_cloud(publish = True) #converts depth image into point cloud
        # plane = self.scene_parser.get_plane(publish = True)
        # exit()

        while not rospy.is_shutdown():
            if self.state == States.IDLE:
                self.state = States.VISUAL_SERVOING
                if objectManipulationState == States.PLACE:
                    self.send_feedback({'msg' : "Trigger Request received. Placing"})
                    self.state = States.PLACE
                else:
                    self.send_feedback({'msg' : "Trigger Request received. Starting to find the object"})
            elif self.state == States.VISUAL_SERVOING:
                success = self.visualServoing.main()
                if success:
                    self.send_feedback({'msg' : "Servoing succeeded! Starting manipulation."})
                    self.state = objectManipulationState
                else:
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
                
                # basic planning here
                self.manipulationMethods.move_to_pregrasp(self.trajectoryClient)
                ee_pose = self.manipulationMethods.getEndEffectorPose()
                self.visualServoing.alignObjectHorizontal(ee_pose_x = ee_pose[0], debug_print = {"ee_pose" : ee_pose})

                self.scene_parser.set_point_cloud(publish = True) #converts depth image into point cloud
                grasp = self.scene_parser.get_grasp(publish = True)
                plane = self.scene_parser.get_plane(publish = True)
                
                
                if grasp:
                    grasp_center, grasp_yaw = grasp
                    
                    if plane:
                        plane_height = plane[1]
                        self.heightOfObject = abs(grasp_center[2] - plane_height)
                    else:
                        self.heightOfObject = 0.2 #default height of object if plane is not detected
                    
                    offsets = self.offset_dict[self.label2name[goal.objectId]]
                    offsets[1] -= 0.01 #constant safety factor
                    grasp = (grasp_center + np.array(offsets)), grasp_yaw
                    self.manipulationMethods.pick(
                        self.trajectoryClient, 
                        grasp,
                        moveUntilContact = self.isContactDict[self.label2name[goal.objectId]]
                    ) 
                    
                    success = self.manipulationMethods.checkIfGraspSucceeded()
                    if success:
                        self.send_feedback({'msg' : "Pick succeeded! "})
                        self.state = States.COMPLETE
                    else:
                        self.send_feedback({'msg' : "Pick failed. Reattempting."})
                        if nPickTriesAttempted >= nPickTriesAllowed:
                            self.send_feedback({'msg' : "Pick failed. Cannot grasp successfully. Aborting."})
                            self.reset()
                            return TriggerResult(success = False)
                        
                        self.send_feedback({'msg' : "Picking failed. Reattempting pick, try number " + str(nPickTriesAttempted) + " of " + str(nPickTriesAllowed) + " allowed."})
                        nPickTriesAttempted += 1
                        self.stow_robot_service()
                        self.state = States.VISUAL_SERVOING

            elif self.state == States.PLACE:
                self.heightOfObject = goal.heightOfObject
                move_to_pose(
                    self.trajectoryClient,
                    {
                        'head_pan;to' : -np.pi/2,
                        'head_tilt;to' : - 30 * np.pi/180,
                        'base_rotate;by': np.pi/2,
                    }
                )
                rospy.sleep(5)
                self.scene_parser.set_point_cloud(publish = True) #converts depth image into point cloud
                plane = self.scene_parser.get_plane(publish = True)
                if plane:
                    # placingLocation = self.scene_parser.get_placing_location(plane, self.heightOfObject, publish = True)
                    placingLocation = np.array([
                        0.0, 0.7, 0.9
                    ])
                    self.manipulationMethods.place(self.trajectoryClient, placingLocation)
                    self.stow_robot_service()
                self.state = States.COMPLETE

            elif self.state == States.COMPLETE:
                # self.send_feedback({'msg' : "Work complete successfully."})
                rospy.loginfo(f"{rospy.get_name()} : Work complete successfully.")
                move_to_pose(self.trajectoryClient, {
                    "head_pan;to" : 0,
                })
                break

        
        self.reset()
        self.server.set_succeeded(result = TriggerResult(success = True, heightOfObject = self.heightOfObject))



if __name__ == '__main__':
    node = ManipulationFSM()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

