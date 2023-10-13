import numpy as np
import rospy
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from nav_msgs.msg import Odometry
import actionlib
import time
import tf
from geometry_msgs.msg import Twist, Vector3

def convertPointToDepth(depth_image, point, intrinsics):
    '''
    depth_image: 2D image that
    point: 2D numpy array of points in the image
    intrinsics: dictionary of camera intrinsics
    '''
    point = point.astype(int)
    point[0] = np.clip(point[0], 0, depth_image.shape[1] - 1)
    point[1] = np.clip(point[1], 0, depth_image.shape[0] - 1)
    
    z = depth_image[point[1], point[0]].astype(float)/1000.0
    x = (point[1] - intrinsics['ppx'])/intrinsics['fx']
    y = (point[0] - intrinsics['ppy'])/intrinsics['fy']
    
    threeDPoint = np.array([x, y, z])
    return threeDPoint

def diff(targetA, sourceA):
    a = targetA - sourceA
    a = np.mod((a + np.pi),2*np.pi) - np.pi
    return a
odometry = None

def odom(data):
    global odometry
    odometry = data

def move_to_pose(trajectoryClient, pose, asynchronous=False, custom_contact_thresholds=False):
    joint_names = [key for key in pose]
    point = JointTrajectoryPoint()
    point.time_from_start = rospy.Duration(0.0)

    trajectory_goal = FollowJointTrajectoryGoal()
    trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
    trajectory_goal.trajectory.joint_names = joint_names
    if not custom_contact_thresholds: 
        joint_positions = [pose[key] for key in joint_names]
        point.positions = joint_positions
        point.effort = []
        trajectory_goal.trajectory.points = [point]
    else:
        pose_correct = all([len(pose[key])==2 for key in joint_names])
        if not pose_correct:
            rospy.logerr("[ALFRED].move_to_pose: Not sending trajectory due to improper pose. custom_contact_thresholds requires 2 values (pose_target, contact_threshold_effort) for each joint name, but pose = {0}".format(pose))
            return
        joint_positions = [pose[key][0] for key in joint_names]
        joint_efforts = [pose[key][1] for key in joint_names]
        point.positions = joint_positions
        point.effort = joint_efforts
        trajectory_goal.trajectory.points = [point]
    trajectory_goal.trajectory.header.stamp = rospy.Time.now()
    trajectoryClient.send_goal(trajectory_goal)
    if not asynchronous: 
        trajectoryClient.wait_for_result()
        #print('Received the following result:')
        #print(self.trajectory_client.get_result())



if __name__ == "__main__":
    # unit tests
    img = np.random.rand(500, 500)
    vals= convertPointToDepth(img, np.array([[1,2],[3,4]]), {'fx': 1, 'fy': 1, 'ppx': 1, 'ppy': 1})
    # point[2] = depth
    print(vals)