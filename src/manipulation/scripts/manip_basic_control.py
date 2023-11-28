#!/usr/bin/env python3


import time
import numpy as np
from tf import TransformerROS
from tf import TransformListener, TransformBroadcaster

from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/planner')

# ROS action finite state machine
import std_msgs
import rospy
import actionlib
import json
# import action message types
from helpers import move_to_pose
import tf
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction
from geometry_msgs.msg import PointStamped

from planner.astar_planner import AStarPlanner
from planner.naive_planner import NaivePlanner

class ManipulationMethods:
    """Methods for object manipulation
    """

    def __init__(self, trajectory_client):
        # rospy.init_node("manipulation_methods")
        self.trajectory_client = trajectory_client
        self.listener : tf.TransformListener = tf.TransformListener()
        self.grip_dist_sub  = rospy.Subscriber('/manipulation/distance', std_msgs.msg.Int32, self.get_grip_dist)
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
        self.grip_dist=None
        self.contact_threshold = {
            'joint_lift' : 7,
            'joint_arm' : 13,
        }
        self.av_effort_window_size = 3
        
        self.states = {
            'joint_lift' : [0 for i in range(self.av_effort_window_size)],
            'joint_arm' : [0 for i in range(self.av_effort_window_size)],
        }
        self.efforts =  {
            'joint_lift' : [0 for i in range(self.av_effort_window_size)],
            'joint_arm' : [0 for i in range(self.av_effort_window_size)],
        }

        rospy.Subscriber('/alfred/joint_states', JointState, self.callback)
        self.cur_robot_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    
    def get_transform(self, base_frame, camera_frame):

        while not rospy.is_shutdown():
            try:
                t = self.listener.getLatestCommonTime(base_frame, camera_frame)
                (trans,rot) = self.listener.lookupTransform(base_frame, camera_frame, t)

                trans_mat = np.array(trans).reshape(3, 1)
                rot_mat = R.from_quat(rot).as_matrix()
                transform_mat = np.hstack((rot_mat, trans_mat))

                return transform_mat
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
    
    def callback(self, msg, verbose = False):
        arm_efforts = 0
        for i, k in enumerate(msg.name):
            if k == 'joint_lift':
                self.states[k].append(msg.position[i])
                self.efforts[k].append(msg.effort[i])
                self.states[k].pop(0)
                self.efforts[k].pop(0)
                
                av_effort = np.mean(self.efforts[k])
                if av_effort < self.contact_threshold[k]:
                    self.isContact = True
                    rospy.logdebug("Lift effort is too low: {}, there is contact".format(av_effort))
                else:
                    self.isContact = False
                    
                self.cur_robot_state[3] = msg.position[i]
            
            if 'arm' in k:
                arm_efforts += abs(msg.effort[i])
                
            if 'wrist' in k:
                self.cur_robot_state[4] = msg.position[i]
                
            # if k == "joint_head_pan":
            #     final_state[2] -= msg.position[i]
            
        self.efforts['joint_arm'].append(arm_efforts / 5)
        self.efforts['joint_arm'].pop(0)
        
        av_effort = np.mean(self.efforts['joint_arm'])
        # print("arm avg effort = ", av_effort)
        if av_effort > self.contact_threshold['joint_arm']:
            self.isContact = True
            rospy.logdebug("Arm effort is too high: {}, there is contact".format(av_effort))

        transform = self.get_transform('link_arm_l4', 'link_arm_l0')
        self.states['joint_arm'].append(self.cur_robot_state[5])
        self.cur_robot_state[5] = np.linalg.norm(transform[:3, 3])
        
        if verbose:
            debug_state = {
                "x" : self.cur_robot_state[0],
                "y" : self.cur_robot_state[1],
                "theta" : self.cur_robot_state[2],
                "lift" : self.cur_robot_state[3],
                "wrist_yaw" : self.cur_robot_state[4],
                "extension" : self.cur_robot_state[5],
            }
            print(debug_state)
        
        
    
    def move_to_pregrasp(self):
        
        move_to_pose(self.trajectory_client, {
            # "lift;to" : 0.85,
            "head_pan;to" : -np.pi/2,
            "head_tilt;to" : -30 * np.pi/180,
            # "head_tilt;to" : 0 * np.pi/180
            'arm;to' : 0.02,
        })
        move_to_pose(self.trajectory_client, {
            "stretch_gripper;to" : 100,
        })
        rospy.sleep(2)

    def getEndEffectorPose(self):
        base_point = PointStamped()
        base_point.header.frame_id = '/link_grasp_center'
        base_point.point.x = 0
        base_point.point.y = 0
        base_point.point.z = 0
        point = self.listener.transformPoint('base_link', base_point).point
        return [point.x, point.y, point.z]

        
    def reorient_base(self, angle):
        
        move_to_pose(self.trajectory_client, {
            'base_rotate;by' : angle
        })
        rospy.sleep(2)

    def open_drawer(self, x, y, z):
        trajectory_client = self.trajectory_client
        
        move_to_pose(trajectory_client, {
            "lift;to" : 0.72,
            'wrist_yaw;to' : np.pi/2, 
            "head_pan;to" : -np.pi/2,
            "head_tilt;to" : -30 * np.pi/180,
            'arm;to' : 0.02,
        })
        
        move_to_pose(trajectory_client, {
            "stretch_gripper;to" : -80,
        })
        
        ee_x, ee_y, ee_z = self.getEndEffectorPose()
        move_to_pose(trajectory_client, {
            'arm;by' : abs(y - ee_y) - 0.1,
        })
        self.move_until_contact_arm({
            'arm;to' : 1,
        })
        
        self.move_until_contact_lift({
            'lift;to' : z - 0.1,
        }, move_up = 0.01)
        
        
        move_to_pose(trajectory_client, {
            'arm;by' : -0.4,
        })
        
        move_to_pose(trajectory_client, {
            'lift;by' : 0.1,
        })
        
        move_to_pose(trajectory_client, {
            'arm;to' : 0.01,
            # 'lift;to' : 0.4,
            'wrist_yaw;to' : np.pi
        })
        
        
        return True
    
    
    def close_drawer(self, x):
        move_to_pose(self.trajectory_client, {
            'base_translate;by' : x
        })
        move_to_pose(self.trajectory_client, {
            'arm;to' : 0.01,
        })
        move_to_pose(self.trajectory_client, {
            'lift;to' : 0.66,
            'wrist_yaw;to' : np.pi/2,
        })
        rospy.sleep(2)
        ee_x, ee_y, ee_z = self.getEndEffectorPose()
        move_to_pose(self.trajectory_client, {
            'arm;by' : 0.2,
        })
        self.move_until_contact_arm({
            'arm;to' : 1,
        })
        move_to_pose(self.trajectory_client, {
            'arm;to' : 0.02,
        }
        )
        move_to_pose(self.trajectory_client, {
            "lift;to" : 0.4,
            'wrist_yaw;to' : np.pi
        })
        
        
    def move_until_contact_arm(self, pose):
        """Function to move the robot until contact is made with the table

        Args:
            trajectory_client (actionserver): Client for the trajectory action server
            pose (dict): Dictionary of the pose to move to
        """
        trajectory_client = self.trajectory_client
        curarm = self.states['joint_arm'][-1]
        self.isContact = False
        while not self.isContact or curarm >= pose['arm;to']:
            curarm = self.states['joint_arm'][-1]
            if self.isContact:
                break
            move_to_pose(trajectory_client, {
                'arm;by': 0.01,
            }, asynchronous = False)

        move_to_pose(trajectory_client, { # moving up slightly so that the grasp is accurate
            'arm;by': -0.02,
        })    
    
    def move_until_contact_lift(self, pose, move_up = 0.02):
        """Function to move the robot until contact is made with the table

        Args:
            trajectory_client (actionserver): Client for the trajectory action server
            pose (dict): Dictionary of the pose to move to
        """
        trajectory_client = self.trajectory_client
        curz = self.states['joint_lift'][-1]
        self.isContact = False
        
        while not self.isContact or curz >= pose['lift;to']:
            curz = self.states['joint_lift'][-1]
            if self.isContact:
                break
            move_to_pose(trajectory_client, {
                'lift;by': -0.01,
            })
        move_to_pose(trajectory_client, { # moving up slightly so that the grasp is accurate
            'lift;to': curz + move_up,
        })

    def plan_and_pick(self, grasp, cloud, moveUntilContact = False, demo = False):
        # self.pick(self.trajectory_client, grasp)
        
        # (x, y, z), theta = grasp
        
        # x, y, theta, z, phi, ext
        curstate = self.cur_robot_state
        
        x, y, theta, z, phi, ext= curstate
        ee_x, ee_y, ee_z = self.getEndEffectorPose()
        (x_g, y_g, z_g), grasp_yaw = grasp
        
        offset_z = z - ee_z # difference between lift and the end effector height.
        offset_y = abs(abs(ext) - abs(ee_y))
        rospy.loginfo("Offset z = {}".format(offset_z))
        rospy.loginfo("Offset y = {}".format(offset_y))
        rospy.loginfo("pregrasp location = {}".format((x_g, y_g, z_g)))
        if not moveUntilContact:
            grasp_yaw = 0
            
        goal_state = np.array([
            x + x_g,
            y,
            theta,
            z_g + offset_z,
            grasp_yaw,
            abs(y_g) - 0.3 - 0.2
        ])
        rospy.loginfo("Current state is {}".format(curstate))
        rospy.loginfo("Goal state is {}".format(goal_state))

        # planner = NaivePlanner(curstate, goal_state, cloud)
        # plan, success = planner.plan()
        # rospy.loginfo("Naive planner success = {}".format(success))
        # if success:
        #     self.pick(
        #         grasp,
        #         moveUntilContact
        #     )
        #     return True


        rospy.loginfo("Naive planner failed, using A*")
        planner = AStarPlanner(curstate, goal_state, cloud)
        plan, success = planner.plan()
    
        if success:
            rospy.loginfo("Plan successful. Executing trajectory")
            self.execute_trajectory(
                plan, 
                visualize = True, 
                planner = planner, 
                dummy = demo
            )        
            
            if not demo:
                self.pick(
                    ((0, y_g, z_g), grasp_yaw),
                    moveUntilContact
                )
                
            return True
            
            # return True
        return False
        
    def plan_and_stow(self, grasp, cloud, moveUntilContact = False):
        # (x, y, z), theta = grasp
        # x, y, theta, z, phi, ext
        
        curstate = self.cur_robot_state


        goal_state = np.array([
            curstate[0],
            curstate[1],
            curstate[2],
            0.4,
            np.pi,
            0.0,
        ])
        rospy.loginfo("Current state is {}".format(curstate))
        rospy.loginfo("Goal state is {}".format(goal_state))
        planner = NaivePlanner(curstate, goal_state, cloud)
        plan, success = planner.plan()
        rospy.loginfo("Naive planner success = {}".format(success))
        
        if not success:
            rospy.loginfo("Naive planner failed, using A*")
            planner = AStarPlanner(curstate, goal_state, cloud)
            plan, success = planner.plan()
        
        if success:
            rospy.loginfo("Plan successful. Executing trajectory")
            self.execute_trajectory(self.trajectory_client, plan, visualize = True, planner = planner)        

            return True
        return False

    def pick(self, grasp, moveUntilContact = False):
        """ Called after the pregrasp pose is reached"""

        (x_g, y_g, z_g), grasp_yaw = grasp
        # x_g -= 0.02 # offset from poor gripper calibration.
        
        ee_x, ee_y, ee_z = self.getEndEffectorPose()
        
        if not moveUntilContact:
            move_to_pose(self.trajectory_client, {
                "lift;by": z_g - ee_z,
            })
        else:
            move_to_pose(self.trajectory_client, {
                "lift;to": z_g + 0.1,
            })
            # rospy.sleep(5)
        
        if moveUntilContact:
            move_to_pose(self.trajectory_client, {
                'wrist_yaw;to' : grasp_yaw
            })
        else:
            move_to_pose(self.trajectory_client, {
                'wrist_yaw;to': 0
            })
            
        rospy.sleep(3)
        
        ee_x, ee_y, ee_z = self.getEndEffectorPose()
        
        move_to_pose(self.trajectory_client, {
            'base_translate;by' : x_g - ee_x
        })
        rospy.sleep(3)
        
        # print("Extending to ", x_g, y_g, z_g, grasp_yaw)
        # print("end effector pose ", ee_x, ee_y, ee_z)
            
        rospy.loginfo("EE_pose is {}".format(ee_y))
        rospy.loginfo("Extending to {}".format((grasp)))
        move_to_pose(self.trajectory_client, {
            "arm;by": abs(y_g - ee_y),
        })
        # rospy.sleep(5)

        if moveUntilContact:
            self.move_until_contact_lift(
                {
                    "lift;to": z_g - 0.2,
                }
            )
        
        move_to_pose(self.trajectory_client, {
            "stretch_gripper;to": -80,
        })
        rospy.sleep(3)

        move_to_pose(self.trajectory_client, {
            "lift;by": 0.1,
            "arm;to" : 0.02,
        })
        
        # rospy.sleep(5)

        # move_to_pose(self.trajectory_client, {
        #     "lift;to": 0.4,
        #     "head_pan;to": 0,
        # })
        # rospy.sleep(5)

            
    def place(self, placing_location):
        """Placing the object at the desired location

        Args:
            self.trajectory_client (): trajectory client
            x (foat): x position of the location to place
            y (float): y position of the location to place
            z (float): z position of the location to place
            theta (_type_): _description_
        """
        x, y, z = placing_location
        
        ee_x, ee_y, ee_z = self.getEndEffectorPose()
        
        move_to_pose(self.trajectory_client, {
            "lift;by": z - ee_z,
        })
        # rospy.sleep(3)
        
        move_to_pose(self.trajectory_client, {
            "wrist_yaw;to": 0,
        })
        rospy.sleep(3)
        ee_x, ee_y, ee_z = self.getEndEffectorPose()
        
        
        # move_to_pose(self.trajectory_client, {
        #     'base_translate;by' : (x - ee_x)
        #     }
        # )
        # rospy.sleep(3)
        
        
        rospy.loginfo("EE_pose is {}".format(ee_y))
        rospy.loginfo("Extending by {}".format(y - ee_y))
        rospy.loginfo("Placing location is {}".format(y))
        move_to_pose(self.trajectory_client, {
            "arm;by": abs(y - ee_y),
            }
        )
        # rospy.sleep(5)
        self.move_until_contact_lift(
            {
                "lift;to": z - 0.2,
            }
        )
        move_to_pose(self.trajectory_client, {
            "stretch_gripper;to": 100,
        })
        rospy.sleep(3)
        
        move_to_pose(self.trajectory_client, {
            'lift;by' : 0.1,
            "arm;to": 0.02,
        })

        # rospy.sleep(5)
        move_to_pose(self.trajectory_client, {
            "wrist_yaw;to": np.pi,
        })
        rospy.sleep(2)
        
        # move_to_pose(self.trajectory_client, {
        #     "lift;to": 0.4,
        # })
        return True
        
    def visualize_boxes_rviz(self, boxes, centers):
        colors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        for i, box in enumerate(boxes):
            center = np.array(box.center)
            extent = np.array(box.extent)
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()

            # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
            marker.type = 1
            marker.id = i

            # Set the scale of the marker
            marker.scale.x = extent[0] 
            marker.scale.y = extent[1]
            marker.scale.z = extent[2]

            # Set the color
            color = colors[i % len(colors)]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.5

            # Set the pose of the marker
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = center[2]
            
            ori = np.array(box.R)
            
            quat = R.from_matrix(ori).as_quat()

            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            self.marker_pub.publish(marker)    
    
    
    def execute_trajectory(self, plan, visualize = False, planner = None, dummy = False):
        """
        takes in a series of waypoints and executes it one by one as the target is reached.
        """
        if len(plan) == 0:
    
            return
        # x, y, theta, z, phi, ext
        waits = [0.5, 0.1, 0.5, 0.5, 0.1, 0.5]
        for i, waypoint in enumerate(plan):
            
            if visualize:
                if planner is None:
                    return 
                boxes, centers = planner.get_collision_boxes(waypoint)
                self.visualize_boxes_rviz(boxes, centers)     
            idx_to_move = 0
            for j in range(len(waypoint)):
                if waypoint[j] != plan[i-1][j]:
                    idx_to_move = j
                    break
            rospy.loginfo("Moving to waypoint {}".format(waypoint))
            final_waypoint = waypoint.copy()
            if idx_to_move == 0 and i != 0:
                rospy.loginfo("waypoint[0] = {}".format(waypoint[0]))
                rospy.loginfo("plan[i] = {}".format(plan[i][0]))
                rospy.loginfo("plan[i-1] = {}".format(plan[i-1][0]))
                final_waypoint[0] = final_waypoint[0] - plan[i-1][0]
                
            if not dummy:
                if idx_to_move == 0:
                    move_to_pose(self.trajectory_client, {
                        "base_translate;by" : final_waypoint[0],
                    })
                else:
                    move_to_pose(self.trajectory_client, {
                        "lift|;to" : final_waypoint[3],
                        "arm|;to" : final_waypoint[5],
                        "wrist_yaw|;to": final_waypoint[4],
                    })
            rospy.sleep(waits[idx_to_move])

        if not dummy:
            move_to_pose(self.trajectory_client, {
                "lift|;to" : waypoint[3],
                "arm|;to" : waypoint[5],
                "wrist_yaw|;to": waypoint[4],
            })
    def pick_with_feedback(self, x, y, z, theta):
        """
        DEPRECATED
        Uses laser TOF sensor to pick up object
        """
        print("Extending to ", x, y, z, theta)
        print("EXECUTING GRASP")

        move_to_pose(self.trajectory_client, {
            "wrist_yaw;to" : 180,
        })

        rospy.sleep(3)
        
        move_to_pose(self.trajectory_client, {
            "lift;to": z,
        })
        rospy.sleep(5)
        move_to_pose(self.trajectory_client, {
            "stretch_gripper;to": 100,
        })

        move_to_pose(self.trajectory_client, {
            "wrist_yaw;to" : 0,
        })
        rospy.sleep(3)

        move_to_pose(self.trajectory_client, {
            "arm;to": y-0.1,
        })
        rospy.sleep(7)

        mount_angle= np.pi/4
        mount_height = 0.025
        offset_gripper  = 0.12 #measured
        object_width = 0.03

        #equation when object is mounted 6cm above gripper abd 12cm away from gripper center
        step = mount_height*np.sin(mount_angle) + ((self.grip_dist/1000.)  - offset_gripper)*np.cos(mount_angle) + object_width

        print(" y - 10 reached, moving to ", step)

        move_to_pose(self.trajectory_client, {
            "arm;by": step,
        })
        rospy.sleep(4)

        print("Alignment complete")
        
        move_to_pose(self.trajectory_client, {
            "stretch_gripper;to": -40,
        })

        rospy.sleep(5)
        move_to_pose(self.trajectory_client, {
            "lift;to": z + 0.10,
        })
        rospy.sleep(3)
# 
        move_to_pose(self.trajectory_client, {
           "arm;to" : 0,
           
        })
        move_to_pose(self.trajectory_client, {
            "wrist_yaw;to" : np.pi,
           
        })
        rospy.sleep(5)
        move_to_pose(self.trajectory_client, {
            "lift;to": z - 0.3,
        })
        rospy.sleep(3)

    def get_grip_dist(self,msg):
        """Subscribes to the /manipulation/distance topic and stores the gripper distance from object"""

        print("reading distance",msg.data)
        self.grip_dist = msg.data
        
    
if __name__=='__main__':
    # test case

    x = 0 # If x!=0 , then robot needs to be translated to be in line with the object of interest
    y = 0.8 # Arm extension
    z = 0.84 # Arm going up # Absolute value of the robot state wrt base # Maxes out at 1.10
    theta = 0 # Yaw, 0 for pointing towards the object
    rospy.init_node("manip", anonymous=False) 
    trajectory_client = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    trajectory_client.wait_for_server()
    obj = ManipulationMethods()

    y = abs(abs(y) - obj.getEndEffectorPose()[0])
    obj.pick_with_feedback(trajectory_client, 0, y, z, theta)

