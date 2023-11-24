#!/usr/local/lib/robot_env/bin/python3

# ROS imports
import actionlib
import rospkg
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from control_msgs.msg import FollowJointTrajectoryAction
import threading
import concurrent.futures

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionFeedback, MoveBaseFeedback, MoveBaseActionResult, MoveBaseResult
from alfred_navigation.msg import NavManAction, NavManGoal, NavManResult, NavManFeedback
from moveback_recovery.msg import MovebackRecoveryAction, MovebackRecoveryGoal
from utils import get_quaternion
from costmap_callback import ProcessCostmap
import time

class NavigationManager():

    # _feedback = alfred_navigation.msg.NavManFeedback()
    # _result = alfred_navigation.msg.NavManResult()


    def __init__(self):
        
        self.node_name = 'navigation_manager'
        rospy.init_node(self.node_name)
        
        self.goal = None

        self.nav_result = None

        self.navman_server = actionlib.SimpleActionServer('nav_man', NavManAction, auto_start = False)
        self.movebase_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        self.navman_server.register_preempt_callback(self.pre_emption)
        self.navman_server.register_goal_callback(self.goal_cb)
        

        self.costmap_processor = ProcessCostmap(self.pre_emption)
        
        # self.moveback_client = actionlib.SimpleActionClient('moveback_recovery', MovebackRecoveryAction)

        rospy.loginfo(f"[{self.node_name}]:" + "Waiting for move_base server...")
        self.movebase_client.wait_for_server()

        self.n_max_attempts = rospy.get_param("/navigation/n_attempts", 3)
        self.n_attempts = 0
        self.last_update_time = time.time()
        
        self.navigating = False
        self.preempted = False
        self.navman_server.start()
        rospy.loginfo(f"[{self.node_name}]:" + "Navigation Manager is ready!")

    def goal_cb(self):
        self.goal = self.navman_server.accept_new_goal()
        self.movebase_client.cancel_all_goals()
        rospy.loginfo(f"[{self.node_name}]:" + "Cancelled movebase. Received new goal to navigate to ({}, {})".format(self.goal.x, self.goal.y))
        
    def pre_emption(self):
        rospy.loginfo(f"[{self.node_name}]:" + "Pre-empting goal...")
        self.preempted = True
        self.navman_server.set_preempted()
        
        
    def main(self):
        while not rospy.is_shutdown():
            if self.goal is not None:
                self.navigate_to_goal(self.goal)
                self.goal = None
            else:
                rospy.sleep(0.1)

    def navigate_to_goal(self, goal):
        rospy.loginfo(f"[{self.node_name}]:" + "Navigation Manager received goal")
        x, y, z = self.costmap_processor.findNearestSafePoint(goal.x, goal.y, 0.0)

        goal.x = x
        goal.y = y

        self.goal = goal
        self.costmap_processor.set_goal(goal)

        # navigate to goal location
        movebase_goal = MoveBaseGoal()
        movebase_goal.target_pose.header.frame_id = "map"
        movebase_goal.target_pose.pose.position.x = self.goal.x
        movebase_goal.target_pose.pose.position.y = self.goal.y
        quaternion = get_quaternion(self.goal.theta)
        movebase_goal.target_pose.pose.orientation = quaternion

        self.movebase_client.send_goal(movebase_goal, feedback_cb = self.navigation_feedback)
        rospy.loginfo(f"[{self.node_name}]:" + "Sending goal to move_base")
        
        wait = self.movebase_client.wait_for_result()
        
        if self.movebase_client.get_state() == actionlib.GoalStatus.PREEMPTED:
            rospy.loginfo(f"[{self.node_name}]:" +"Preempted")
            self.navigate_to_goal(self.goal)
        elif self.movebase_client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[{self.node_name}]:" +"Failed to reach goal. Debugging...")
            self.debug_nav_failure()
        else:
            self.send_navman_result(success = True)
        

    
    def debug_nav_failure(self):
        if self.n_attempts < self.n_max_attempts:
            self.n_attempts += 1
            rospy.loginfo(f"[{self.node_name}]:" + f"Trying again: {self.n_max_attempts - self.n_attempts} left")
            rospy.loginfo(f"Have attempted " + str(self.n_attempts) + " times")
            self.navigate_to_goal(self.goal)
        else:
            self.send_navman_result(success = False)
            
        
        
    def send_navman_result(self, success):
        self.costmap_processor.set_goal(None)
        navman_result = NavManResult()
        self.goal = None
        self.n_attempts = 0
        if success:
            navman_result.success = True
            self.navman_server.set_succeeded(navman_result)
            rospy.loginfo(f"[{self.node_name}]:" + f"Goal reached successfully.")
        else:
            navman_result.success = False
            self.navman_server.set_aborted(navman_result)
            rospy.loginfo(f"[{self.node_name}]:" + f"Goal could not be reached.")


    def navigation_feedback(self, msg : MoveBaseFeedback):
        self.position = msg.base_position.pose.position
        self.orientation = msg.base_position.pose.orientation
        
        # if time.time() - self.last_update_time > 1:
        #     self.last_update_time = time.time()
        #     rospy.loginfo(f"[{self.node_name}]:" +"Current position: {}".format(self.position))
        #     rospy.loginfo(f"[{self.node_name}]:" +"Current orientation: {}".format(self.orientation))
        


if __name__ == '__main__':

    
    node = NavigationManager()
    node.main()
        