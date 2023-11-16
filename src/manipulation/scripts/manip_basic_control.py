#!/usr/bin/env python3

# TODO:
# Add feedbacks (debug)
# placing plane detection.
# Is grasp complete?

import time
import numpy as np
from tf import TransformerROS
from tf import TransformListener, TransformBroadcaster
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


class ManipulationMethods:
    """Methods for object manipulation
    """

    def __init__(self):
        # rospy.init_node("manipulation_methods")
        self.listener : tf.TransformListener = tf.TransformListener()
        self.grip_dist_sub  = rospy.Subscriber('/manipulation/distance', std_msgs.msg.Int32, self.get_grip_dist)

        # deprecated since servo is not required visual alignment now
        # self.grip_servo_pub = rospy.Publisher('/manipulation/servo', std_msgs.msg.UInt16, queue_size=10)
        # self.grip_servo_pub.publish(178)

        self.grip_dist=None

        self.contact_threshold = 7


        self.av_effort = 20
        self.av_effort_window_size = 3
        self.states = [0 for i in range(self.av_effort_window_size)]
        self.efforts = [0 for i in range(self.av_effort_window_size)]

        rospy.Subscriber('/alfred/joint_states', JointState, self.callback)
        
    def callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] == 'joint_lift':
                effort = msg.effort[i]
                self.states.append(msg.position[i])
                self.efforts.append(msg.effort[i])
                self.states.pop(0)
                self.efforts.pop(0)
                
                av_effort = np.mean(self.efforts)
                if av_effort < self.contact_threshold:
                    self.isContact = True
                    print("Lift effort is too low: {}, there is contact".format(av_effort))
                else:
                    self.isContact = False
    
    def move_to_pregrasp(self, trajectoryClient):
        # #Changed yaw to 180 since sometimes gripper will collide with table at yaw=0
        move_to_pose(trajectoryClient, {
            "lift;to" : 0.85,
            "head_pan;to" : -np.pi/2,
            "head_tilt;to" : -30 * np.pi/180
            # "head_tilt;to" : 0 * np.pi/180
            })
        move_to_pose(trajectoryClient, {
            "wrist_yaw;to" : 0,
            "stretch_gripper;to" : 100,
        })
        rospy.sleep(5)

        
    def getEndEffectorPose(self):
        base_point = PointStamped()
        base_point.header.frame_id = '/link_grasp_center'
        base_point.point.x = 0
        base_point.point.y = 0
        base_point.point.z = 0
        point = self.listener.transformPoint('base_link', base_point).point
        return [point.x, point.y, point.z]


    # def getEndEffectorPose(self):
    #     from_frame_rel = 'link_grasp_center'
    #     to_frame_rel = 'base_link'
    #     while not rospy.is_shutdown():
    #         try:
    #             translation, rotation = self.listener.lookupTransform(to_frame_rel, from_frame_rel, rospy.Time(0))
    #             rospy.loginfo('The pose of target frame %s with respect to %s is: \n %s, %s', to_frame_rel, from_frame_rel, translation, rotation)
    #             return [(translation[0]), translation[1], translation[2]]
    #         except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #             continue
            
    #     rospy.logerr(f"[{rospy.get_name()}] " + "Transform not found between {} and {}".format(to_frame_rel, from_frame_rel))
    #     return None
    
       
    def get_grip_dist(self,msg):
        """Subscribes to the /manipulation/distance topic and stores the gripper distance from object"""

        print("reading distance",msg.data)
        self.grip_dist = msg.data

    def pick(self, trajectoryClient, x, y, z, theta, moveUntilContact = False):
        """ Called after the pregrasp pose is reached"""

        print("Extending to ", x, y, z, theta)
        print("EXECUTING GRASP")
        
        move_to_pose(trajectoryClient, {
            "lift;to": z,
        })
        move_to_pose(trajectoryClient, {
            "arm;to": y,
        })

        if moveUntilContact:
            self.move_until_contact(
                trajectoryClient,
                {
                    "lift;to": z - 0.2,
                }
            )
        
        move_to_pose(trajectoryClient, {
            "stretch_gripper;to": -80,
        })
        rospy.sleep(3)

        move_to_pose(trajectoryClient, {
            "lift;to": z + 0.10,
            "arm;to" : 0,
            "wrist_yaw;to" : np.pi,
        })

        move_to_pose(trajectoryClient, {
            "lift;to": z - 0.3,
            "head_pan;to": 0,
        })

        
    def pick_with_feedback(self, trajectoryClient, x, y, z, theta):
        """Uses laser TOF sensor to pick up object"""

        print("Extending to ", x, y, z, theta)
        print("EXECUTING GRASP")

        move_to_pose(trajectoryClient, {
            "wrist_yaw;to" : 180,
        })

        rospy.sleep(3)
        
        move_to_pose(trajectoryClient, {
            "lift;to": z,
        })
        rospy.sleep(5)
        move_to_pose(trajectoryClient, {
            "stretch_gripper;to": 100,
        })

        move_to_pose(trajectoryClient, {
            "wrist_yaw;to" : 0,
        })
        rospy.sleep(3)

        move_to_pose(trajectoryClient, {
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

        move_to_pose(trajectoryClient, {
            "arm;by": step,
        })
        rospy.sleep(4)

        print("Alignment complete")
        
        move_to_pose(trajectoryClient, {
            "stretch_gripper;to": -40,
        })

        rospy.sleep(5)
        move_to_pose(trajectoryClient, {
            "lift;to": z + 0.10,
        })
        rospy.sleep(3)
# 
        move_to_pose(trajectoryClient, {
           "arm;to" : 0,
           
        })
        move_to_pose(trajectoryClient, {
            "wrist_yaw;to" : np.pi,
           
        })
        rospy.sleep(5)
        move_to_pose(trajectoryClient, {
            "lift;to": z - 0.3,
        })
        rospy.sleep(3)

    def move_until_contact(self, trajectory_client, pose):
        """Function to move the robot until contact is made with the table

        Args:
            trajectory_client (actionserver): Client for the trajectory action server
            pose (dict): Dictionary of the pose to move to
        """
        while len(self.states) == 0:
            rospy.sleep(0.5)
            
        curz = self.states[-1]
        self.isContact = False
        while not self.isContact or curz >= pose['lift;to']:
            curz = self.states[-1]
            # print(curz, pose['lift;to'], isContact)
            if self.isContact:
                break
            move_to_pose(trajectory_client, {
                'lift;to': curz - 0.01,
            }, asynchronous = True)
            rospy.sleep(0.05)
        move_to_pose(trajectory_client, { # moving up slightly so that the grasp is accurate
            'lift;to': curz + 0.04,
        })

    def checkIfGraspSucceeded(self):
        """Checks if the grasp was successful or not

        Returns:
            bool: True if successful, False otherwise
        """
        return True
            
    def place(self, trajectoryClient, x, y, z, theta):
        """Placing the object at the desired location

        Args:
            trajectoryClient (): trajectory client
            x (foat): x position of the location to place
            y (float): y position of the location to place
            z (float): z position of the location to place
            theta (_type_): _description_
        """

        move_to_pose(trajectoryClient, {
            "lift;to": z,
        })

        rospy.sleep(5)
        move_to_pose(trajectoryClient, {
            "wrist_yaw;to": theta*(np.pi/180),
        })

        rospy.sleep(3)

        move_to_pose(trajectoryClient, {
            "arm;to": y,
        })
        
        rospy.sleep(3)

        self.move_until_contact(
            trajectoryClient,
            {
                "lift;to": z - 0.2,
            }
        )
        move_to_pose(trajectoryClient, {
            "stretch_gripper;to": 50,
        })

        rospy.sleep(4)
        
        move_to_pose(trajectoryClient, {
            "arm;to": 0,
        })
        
        rospy.sleep(0.5)

        move_to_pose(trajectoryClient, {
            "wrist_yaw;to": np.pi,
        })
        rospy.sleep(7)
        
        rospy.sleep(0.5)
        move_to_pose(trajectoryClient, {
            "lift;to": 0.4,
        })
        
        # rospy.sleep(6)
        
    
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


    # try:
    #     rospy.spin()
    # except rospy.ROSInterruptException:
    #     pass
      