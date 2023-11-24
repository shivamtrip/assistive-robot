import numpy as np

import rospy
from alfred_navigation.msg import NavManAction, NavManGoal
from geometry_msgs.msg import PointStamped
import actionlib
rospy.init_node("test_navstack")


def callback(msg): 
    x = msg.point.x
    y = msg.point.y
    z = msg.point.z
    rospy.loginfo("coordinates:x=%f y=%f" %(x,y))
    
    goal = NavManGoal()
    goal.x = x
    goal.y = y
    goal.theta = np.random.uniform(-np.pi, np.pi)
    navigation_client.cancel_goal()
    navigation_client.send_goal(goal)
    
    
rospy.set_param('/navigation/n_attempts', 1)
navigation_client = actionlib.SimpleActionClient('nav_man', NavManAction)
rospy.loginfo(f"[{rospy.get_name()}]:" + "Waiting for Navigation Manager server...")
navigation_client.wait_for_server()

sub = rospy.Subscriber("/clicked_point", PointStamped, callback)

try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")