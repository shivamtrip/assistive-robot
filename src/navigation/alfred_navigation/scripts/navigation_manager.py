#!/usr/local/lib/robot_env/bin/python3

# ROS imports
import actionlib
import rospkg
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from control_msgs.msg import FollowJointTrajectoryAction
import threading
import concurrent.futures
from std_msgs.msg import Bool 

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionFeedback, MoveBaseFeedback, MoveBaseActionResult, MoveBaseResult
from alfred_navigation.msg import NavManAction, NavManGoal, NavManResult, NavManFeedback
from moveback_recovery.msg import MovebackRecoveryAction, MovebackRecoveryGoal
from utils import get_quaternion
import dynamic_reconfigure.client

class NavigationManager():

    # _feedback = alfred_navigation.msg.NavManFeedback()
    # _result = alfred_navigation.msg.NavManResult()


    def __init__(self):

        self.nav_goal_x = None
        self.nav_goal_y = None
        self.nav_goal_theta = None

        self.nav_result = None


        self.low_res_pcd_status_pub = rospy.Publisher("low_res_pcd_status", Bool, queue_size=10)

        self.navman_server = actionlib.SimpleActionServer('nav_man', NavManAction, self.update_goal, False)
        self.movebase_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for move_base server...")
        self.movebase_client.wait_for_server()

        self.global_map_client = dynamic_reconfigure.client.Client("/move_base/global_costmap/rgbd_obstacle_layer", timeout = 10)
        self.local_map_client = dynamic_reconfigure.client.Client("/move_base/local_costmap/rgbd_obstacle_layer", timeout = 10)

        self.navman_server.start()
        print("NavMan Server has started..")
        


    def navigation_dynamic_reconfigure(self, rgbd_layer_status):

        self.global_reconfigure_params = { 'enabled' : rgbd_layer_status, 'publish_voxel_map' : rgbd_layer_status}
        self.local_reconfigure_params = { 'enabled' : rgbd_layer_status, 'publish_voxel_map' : rgbd_layer_status}

        global_map_config = self.global_map_client.update_configuration(self.global_reconfigure_params)
        local_map_config = self.local_map_client.update_configuration(self.local_reconfigure_params)


    def update_goal(self, goal : NavManGoal):
        
        print("Navigation Manager received goal from Task Planner")

        self.low_res_pcd_status_pub.publish(True)

        self.nav_goal_x = goal.x
        self.nav_goal_y = goal.y
        self.nav_goal_theta = goal.theta

        # self.rgbd_layer_status = True
        # print("Navigation Manager will now ENABLE 3D Navigation")
        # self.navigation_dynamic_reconfigure(self.rgbd_layer_status)
        # print("Navigation Manager has enabled 3D Navigation")

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

            print("We have successfully reached the goal!!")
            
            # self.rgbd_layer_status = False
            # print("Navigation Manager will now DISABLE 3D Navigation")
            # self.navigation_dynamic_reconfigure(self.rgbd_layer_status)
            # print("Navigation Manager has disabled 3D Navigation")
            
            self.low_res_pcd_status_pub.publish(False)

            navman_result.success = True
            self.navman_server.set_succeeded(navman_result)
            


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