import numpy as np
import rospy
from helpers import *

rospy.init_node('temp')

trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
trajectoryClient.wait_for_server()


print("sending trajectory")


dic = {
    'lift;to' : (0.8, 40), 
    'arm;to' : (0.5),
    'head_pan;to' : 0.1,
    'head_tilt;to' : 0.1,
    'wrist_yaw;to' : 0.1,
}
move_to_pose(trajectoryClient, dic)

rospy.loginfo("Reached 0.8")


dic = {
    'lift;to' : (0.5, 25), 
    # 'arm;to' : 0.0,
    'head_pan;to' : 0.0,
    'head_tilt;to' : 0.0,
    'wrist_yaw;to' : 0.0,
}
move_to_pose(trajectoryClient, dic )

rospy.loginfo("Reached 0.5")
