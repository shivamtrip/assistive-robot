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


class NavigationManager():

    # _feedback = alfred_navigation.msg.NavManFeedback()
    # _result = alfred_navigation.msg.NavManResult()


    def __init__(self):

        self.nav_goal_x = None
        self.nav_goal_y = None
        self.nav_goal_theta = None

        self.nav_result = None

        self.navman_server = actionlib.SimpleActionServer('nav_man', NavManAction, self.update_goal, False)

        self.movebase_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        # self.moveback_client = actionlib.SimpleActionClient('moveback_recovery', MovebackRecoveryAction)

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for move_base server...")
        self.movebase_client.wait_for_server()

        # self.moveback_client.wait_for_server()

        self.navman_server.start()
        print("NavMan Server has started..")
        

    def update_goal(self, goal : NavManGoal):
        
        print("Navigation Manager received goal from Task Planner")

        self.nav_goal_x = goal.x
        self.nav_goal_y = goal.y
        self.nav_goal_theta = goal.theta

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
        print("Sending goal to move_base")
        # print("Sending goal to move_base", self.goal_locations[locationName]['x'], self.goal_locations[locationName]['y'], self.goal_locations[locationName]['theta'])
             
        wait = self.movebase_client.wait_for_result()

        self.check_result()


    def check_result(self):

        if self.movebase_client.get_state() != actionlib.GoalStatus.SUCCEEDED:

            rospy.loginfo(f"[{rospy.get_name()}]:" +"Failed to reach goal")
            
            # cancel navigation
            self.movebase_client.cancel_goal()

            self.nav_result = False

            self.debug_nav_failure()


        else:
            self.nav_result = True
            self.send_navman_result()

        
    
    def debug_nav_failure(self):

            # print("Sending moveback goal!")
            # moveback_goal = MovebackRecoveryGoal()
            # moveback_goal.execute = True
            # self.moveback_client.send_goal(moveback_goal)

            # print("Successfully sent Moveback goal!")

            print("Sending goal again!")
    
            self.navigate_to_goal()

        
        
    def send_navman_result(self):

        navman_result = NavManResult()

        if self.nav_result:
            navman_result.success = True
            self.navman_server.set_succeeded(navman_result)
            print("We have successfully reached the goal!!")


    def navigation_feedback(self, msg : MoveBaseFeedback):
        self.position = msg.base_position.pose.position
        self.orientation = msg.base_position.pose.orientation
        
        # if time.time() - self.last_update_time > 1:
        #     self.last_update_time = time.time()
        rospy.logdebug(f"[{rospy.get_name()}]:" +"Current position: {}".format(self.position))
        rospy.logdebug(f"[{rospy.get_name()}]:" +"Current orientation: {}".format(self.orientation))



if __name__ == '__main__':

    rospy.init_node('navigation_manager')
    node = NavigationManager()

    rospy.spin()