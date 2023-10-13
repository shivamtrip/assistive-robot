#! /usr/bin/env python3
from __future__ import print_function
import importlib
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryFeedback
from control_msgs.msg import FollowJointTrajectoryResult
import stretch_body.robot as rb
import time
import numpy as np
class JointTrajectoryAction:

    def __init__(self, node, robot):
        self.node = node
        self.robot : rb.Robot = robot
        self.server = actionlib.SimpleActionServer('/alfred_controller/follow_joint_trajectory',
                                                   FollowJointTrajectoryAction,
                                                   execute_cb=self.execute_cb,
                                                   auto_start=False)
        self.feedback = FollowJointTrajectoryFeedback()
        self.result = FollowJointTrajectoryResult()

    def execute_cb(self, goal):
        with self.node.robot_stop_lock:
            self.node.stop_the_robot = False
        # self.node.robot_mode_rwlock.acquire_read()
        commanded_joint_names = goal.trajectory.joint_names
        rospy.loginfo(("[{0}] joint_traj action: New trajectory received with joint_names = "
                       "{1}").format(self.node.node_name, commanded_joint_names))
        # TODO: Add safety constraints with respect to the robot state.

        for name in goal.trajectory.joint_names:
            joint, motionType = name.split(';')
            pos = goal.trajectory.points[0].positions[0]
            # threshold = goal.trajectory.points[0].effort[0]
            if len(goal.trajectory.points[0].effort) > 0:
                threshold = goal.trajectory.points[0].effort[0]
            else:
                threshold = 100
            if 'head' in joint:
                if motionType == 'to':
                    self.robot.head.move_to(joint, pos)
                else:
                    self.robot.head.move_by(joint, pos)
            elif 'base_translate' in joint:
                if motionType == 'to':
                    self.robot.base.move_to(pos)
                elif motionType == 'vel':
                    self.robot.base.set_translate_velocity(pos)
                elif motionType == 'by':
                    self.robot.base.move_by(pos)
            elif 'base_rotate' in joint:
                if motionType == 'to':
                    self.robot.base.rotate_to(pos)
                elif motionType == 'vel':
                    self.robot.base.set_rotational_velocity(pos)
                elif motionType == 'by':
                    self.robot.base.rotate_by(pos)

            elif 'lift' in joint:
                if motionType == 'to':
                    self.robot.lift.move_to(pos, contact_thresh_pos=threshold, contact_thresh_neg=-threshold)
                elif motionType == 'vel':
                    self.robot.lift.set_velocity(pos)
                elif motionType == 'by':
                    self.robot.lift.move_by(pos, contact_thresh_pos=threshold, contact_thresh_neg=-threshold)
            elif 'arm' in joint:
                if motionType == 'to':
                    self.robot.arm.move_to(pos, contact_thresh_pos=threshold, contact_thresh_neg=-threshold)
                elif motionType == 'vel':
                    self.robot.arm.set_velocity(pos)
                elif motionType == 'by':
                    self.robot.arm.move_by(pos, contact_thresh_pos=threshold, contact_thresh_neg=-threshold)
                    
            elif 'stretch_gripper' in joint:
                if motionType == 'to':
                    self.robot.end_of_arm.move_to("stretch_gripper", pos) 
            elif 'wrist_yaw' in joint:
                if motionType == 'to':
                    self.robot.end_of_arm.move_to("wrist_yaw", pos * np.pi/180)  
        self.robot.push_command()

        # self.robot.arm.wait_until_at_setpoint()
        # self.robot.lift.wait_until_at_setpoint()
        # while self.robot.end_of_arm.motors['wrist_yaw'].motor.is_moving():
        #     time.sleep(0.1)
        # while self.robot.end_of_arm.motors['stretch_gripper'].motor.is_moving():
        #     time.sleep(0.1)
            
        self.success_callback("set command.")
        # self.robot_mode_rwlock.release_read()
        

    def invalid_joints_callback(self, err_str):
        if self.server.is_active() or self.server.is_preempt_requested():
            rospy.logerr("{0} joint_traj action: {1}".format(self.node.node_name, err_str))
            self.result.error_code = self.result.INVALID_JOINTS
            self.result.error_string = err_str
            self.server.set_aborted(self.result)

    def invalid_goal_callback(self, err_str):
        if self.server.is_active() or self.server.is_preempt_requested():
            rospy.logerr("{0} joint_traj action: {1}".format(self.node.node_name, err_str))
            self.result.error_code = self.result.INVALID_GOAL
            self.result.error_string = err_str
            self.server.set_aborted(self.result)

    def goal_tolerance_violated_callback(self, err_str):
        if self.server.is_active() or self.server.is_preempt_requested():
            rospy.logerr("{0} joint_traj action: {1}".format(self.node.node_name, err_str))
            self.result.error_code = self.result.GOAL_TOLERANCE_VIOLATED
            self.result.error_string = err_str
            self.server.set_aborted(self.result)

    def success_callback(self, success_str):
        # rospy.loginfo("{0} joint_traj action: {1}".format(self.node.node_name, success_str))
        self.result.error_code = self.result.SUCCESSFUL
        self.result.error_string = success_str
        self.server.set_succeeded(self.result)
