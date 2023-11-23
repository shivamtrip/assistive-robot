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

armclient = actionlib.SimpleActionClient('/stretch_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
diffdrive = rospy.Publisher('/stretch_diff_drive_controller/cmd_vel', Twist, queue_size=0)
headclient = actionlib.SimpleActionClient('/stretch_head_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
gripperclient = actionlib.SimpleActionClient('/stretch_gripper_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

def move_to_pose(trajectoryClient, pose, asynchronous = False):
    
    trajectory_goal = FollowJointTrajectoryGoal()
    trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
    point = JointTrajectoryPoint()
    point.time_from_start = rospy.Duration(0.1)
    
    for k in pose.keys():
        if 'head' in k:
            print("Head pan", pose[k])
            joint, isRel = k.split(';')
            joint = 'joint_' + joint
            joint_positions = [pose[k]]
            point.positions = joint_positions
            point.time_from_start = rospy.Duration(0.2)
            trajectory_goal.trajectory.joint_names = [joint]
            trajectory_goal.trajectory.points = [point]
            headclient.send_goal(trajectory_goal)
        if 'base_rotate' in k:
            rospy.Subscriber('/stretch_diff_drive_controller/odom', Odometry, odom)
            print("Rotating base", pose[k])
            joint, isRel = k.split(';')
            if isRel == 'by':
                while odometry is None:
                    print("Awaiting odometry")
                    continue
                q = odometry.pose.pose.orientation
                r, p, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
                target = pose[k]  
                print("Target", target)
                print("Current", yaw)
                if target > 0:
                    vel = 1
                else:
                    vel = -1
                target += yaw
                while True:
                    q = odometry.pose.pose.orientation
                    r, p, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
                    diffdrive.publish(Twist(
                        linear = Vector3(0, 0, 0),
                        angular = Vector3(0, 0, vel)
                    ))
                    if abs(diff(target, yaw)) < 0.15:
                        break
                diffdrive.publish(Twist(
                    linear = Vector3(0, 0, 0),
                    angular = Vector3(0.0, 0, 0)
                ))
        if 'base_translate' in k:
            joint, isRel = k.split(';')
            diffdrive.publish(Twist(
                linear = Vector3(pose[k], 0, 0),
                angular = Vector3(0, 0, 0)
            ))
        if 'lift' in k:
            print("Lift command received", pose[k])
            joint, isRel = k.split(';')
            joint = 'joint_lift'
            
            joint_positions = [pose[k]]
            point.time_from_start = rospy.Duration(0.2)
            point.positions = joint_positions
            trajectory_goal.trajectory.joint_names = [joint]
            trajectory_goal.trajectory.points = [point]
            trajectory_goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
            armclient.send_goal(trajectory_goal)
            armclient.wait_for_result()

        if 'arm' in k:
            print("Arm command received", pose[k])
            joint, isRel = k.split(';')
            positions = [0, 0, 0, 0]
            total_extension = pose[k]
            maxdist = [0.15, 0.15,0.15,0.15]
            joint_ext = [0] * len(maxdist)
            for i in range(len(maxdist)):
                # If the total extension is less than the maximum extension of the joint, use the total extension
                if total_extension <= maxdist[i]:
                    joint_ext[i] = total_extension
                    break
                else:
                    joint_ext[i] = maxdist[i]
                    total_extension -= maxdist[i]
            
            for i, joint in enumerate(['joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3']):
                joint_positions = [joint_ext[i]]
                point.time_from_start = rospy.Duration(0.2)
                point.positions = joint_positions
                trajectory_goal.trajectory.joint_names = [joint]
                trajectory_goal.trajectory.points = [point]
                trajectory_goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
                armclient.send_goal(trajectory_goal)
            armclient.wait_for_result()
        if 'gripper' in k:
            print("Gripper command received", pose[k])
            joint, isRel = k.split(';')
            joint = 'joint_gripper_finger_left'
            joint_positions = [pose[k]]
            point.positions = joint_positions
            trajectory_goal.trajectory.joint_names = [joint]
            trajectory_goal.trajectory.points = [point]
            gripperclient.send_goal(trajectory_goal)
            joint = 'joint_gripper_finger_right'
            joint_positions = [pose[k]]
            point.positions = joint_positions
            trajectory_goal.trajectory.joint_names = [joint]
            trajectory_goal.trajectory.points = [point]
            gripperclient.send_goal(trajectory_goal)
            gripperclient.wait_for_result()

        if 'yaw' in k:
            print("Yaw command received", pose[k])
            joint, isRel = k.split(';')
            joint = 'joint_wrist_yaw'
            joint_positions = [pose[k]]
            point.positions = joint_positions
            trajectory_goal.trajectory.joint_names = [joint]
            trajectory_goal.trajectory.points = [point]
            armclient.send_goal(trajectory_goal)
            armclient.wait_for_result()
            # yaw doesnt work
    print("Command completed")

if __name__ == "__main__":
    # unit tests
    img = np.random.rand(500, 500)
    vals= convertPointToDepth(img, np.array([[1,2],[3,4]]), {'fx': 1, 'fy': 1, 'ppx': 1, 'ppy': 1})
    # point[2] = depth
    print(vals)