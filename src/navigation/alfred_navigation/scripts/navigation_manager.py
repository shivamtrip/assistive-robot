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
        
        self.nav_goal_x = None
        self.nav_goal_y = None
        self.nav_goal_theta = None

        self.nav_result = None

        self.navman_server = actionlib.SimpleActionServer('nav_man', NavManAction, self.update_goal, False)
        self.movebase_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        self.costmap_processor = ProcessCostmap()
        
        # self.moveback_client = actionlib.SimpleActionClient('moveback_recovery', MovebackRecoveryAction)

        rospy.loginfo(f"[{self.node_name}]:" + "Waiting for move_base server...")
        self.movebase_client.wait_for_server()

        # self.moveback_client.wait_for_server()
        self.n_max_attempts = rospy.get_param("/navman/n_attempts", 3)
        self.n_attempts = 0
        self.last_update_time = time.time()
        
        self.navman_server.start()
        rospy.loginfo(f"[{self.node_name}]:" + "Navigation Manager is ready!")


    def update_goal(self, goal : NavManGoal):
        
        rospy.loginfo(f"[{self.node_name}]:" + "Navigation Manager received goal from Task Planner")

        x, y, z = self.costmap_processor.findNearestSafePoint(goal.x, goal.y, 0.0)
        
        self.nav_goal_x = x
        self.nav_goal_y = y
        self.nav_goal_theta = goal.theta
        self.n_attempts = 0
        
        self.navigate_to_goal()


    def navigate_to_goal(self):        

        movebase_goal = MoveBaseGoal()
        
        movebase_goal.target_pose.header.frame_id = "map"

        # send goal
        movebase_goal.target_pose.pose.position.x = self.nav_goal_x
        movebase_goal.target_pose.pose.position.y = self.nav_goal_y        
        quaternion = get_quaternion(self.nav_goal_theta)
        movebase_goal.target_pose.pose.orientation = quaternion

        self.movebase_client.send_goal(movebase_goal, feedback_cb = self.navigation_feedback)
        
        rospy.loginfo(f"[{self.node_name}]:" + "Sending goal to move_base")

        wait = self.movebase_client.wait_for_result()

        self.check_result()


    def check_result(self):

        if self.movebase_client.get_state() != actionlib.GoalStatus.SUCCEEDED:

            rospy.loginfo(f"[{self.node_name}]:" +"Failed to reach goal. Debugging...")
            self.movebase_client.cancel_goal()
            self.debug_nav_failure()
        else:
            self.send_navman_result(success = True)

        
    
    def debug_nav_failure(self):
        # TODO - in future, add specific conditions where things can be fixed.
        # For now, just try again until success.

        # print("Sending moveback goal!")
        # moveback_goal = MovebackRecoveryGoal()
        # moveback_goal.execute = True
        # self.moveback_client.send_goal(moveback_goal)

        # print("Successfully sent Moveback goal!")

        if self.n_attempts < self.n_max_attempts:
            self.n_attempts += 1
            rospy.loginfo(f"[{self.node_name}]:" + f"Trying again: {self.n_max_attempts - self.n_max_attempts} left")
            self.navigate_to_goal()
        else:
            self.send_navman_result(success = False)
        
        
    def send_navman_result(self, success):

        navman_result = NavManResult()

        if success:
            navman_result.success = True
            self.n_attempts = 0
            self.navman_server.set_succeeded(navman_result)
            rospy.loginfo(f"[{self.node_name}]:" + f"Goal reached successfully.")
        else:
            navman_result.success = False
            self.navman_server.set_aborted(navman_result)
            rospy.loginfo(f"[{self.node_name}]:" + f"Goal could not be reached.")


    def navigation_feedback(self, msg : MoveBaseFeedback):
        self.position = msg.base_position.pose.position
        self.orientation = msg.base_position.pose.orientation
        
        if time.time() - self.last_update_time > 1:
            self.last_update_time = time.time()
            rospy.loginfo(f"[{self.node_name}]:" +"Current position: {}".format(self.position))
            rospy.loginfo(f"[{self.node_name}]:" +"Current orientation: {}".format(self.orientation))
        


if __name__ == '__main__':

    
    node = NavigationManager()

    rospy.spin()