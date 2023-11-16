#!/usr/bin/env python3

import numpy as np
import std_msgs
import rospy
import actionlib
from helpers import move_to_pose
import tf
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction
from std_srvs.srv import Trigger

class ManipulationMethods:

    """Methods for object manipulation"""

    def __init__(self):

        self.listener : tf.TransformListener = tf.TransformListener()
        self.grip_dist_sub  = rospy.Subscriber('/manipulation/distance', std_msgs.msg.Int32, self.get_grip_dist)
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)

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

        move_to_pose(trajectoryClient, {
            "lift;to" : 0.85,
            })

        rospy.sleep(5)

        move_to_pose(trajectoryClient, {
            "wrist_yaw;to" : 0,            
        })

        rospy.sleep(5)
        rospy.loginfo("Pregrasp pose achieved.")

    def getEndEffectorPose(self):

        to_frame_rel = 'base_link'
        from_frame_rel = 'link_grasp_center'
        while not rospy.is_shutdown():
            try:
                translation, rotation = self.listener.lookupTransform(to_frame_rel, from_frame_rel, rospy.Time(0))
                rospy.loginfo('The pose of target frame %s with respect to %s is: \n %s, %s', to_frame_rel, from_frame_rel, translation, rotation)
                return translation
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            
        rospy.logerr(f"[{rospy.get_name()}] " + "Transform not found between {} and {}".format(to_frame_rel, from_frame_rel))
        return None
    
       
    def get_grip_dist(self,msg):
        """Subscribes to the /manipulation/distance topic and stores the gripper distance from object"""

        print("reading distance",msg.data)
        self.grip_dist = msg.data

    def place(self, trajectoryClient, _, y, z, __, moveUntilContact = False, place = False):
        
        # Begin PLACE procedure
        
        move_to_pose(trajectoryClient, {
            "lift;to": z-0.019, # + 0.03 for cm offset from aruco with bottle
        })
        
        rospy.sleep(2)
        
        move_to_pose(trajectoryClient, {
            "arm;to": y - 0.08, # + 0.08 for cm offset from aruco with center of bottle
        })

        rospy.sleep(5)

        move_to_pose(trajectoryClient, {
            "stretch_gripper;to": 100,
        })

        rospy.loginfo("Object release attempted.") 

        rospy.sleep(5)

        move_to_pose(trajectoryClient, {
            "lift;to": z + 0.10,
        })

        # Begin STOW procedure

        move_to_pose(trajectoryClient, {
           "arm;to" : 0,
        })

        move_to_pose(trajectoryClient, {
           "wrist_yaw;to" : np.pi,
        })

        rospy.sleep(10)

        if moveUntilContact:
            
            self.move_until_contact(
                trajectoryClient,
                {
                    "lift;to": z - 0.2,
                }
            )
        
        move_to_pose(trajectoryClient, {
            "head_pan;to": 0,
        }
        )

        rospy.loginfo("Place pipeline complete")
        rospy.sleep(5)

           
    def pick(self, trajectoryClient, x, y, z, theta, moveUntilContact = False, place = False):
        """ Called after the pregrasp pose is reached"""
 
        # Begin PICK procedure

        move_to_pose(trajectoryClient, {
            "lift;to": z+0.03, # have added + 0.03 for cm offset from aruco with bottle
        })

        rospy.sleep(2)

        move_to_pose(trajectoryClient, {
                "stretch_gripper;to": 100,
            })

        rospy.sleep(3)

        move_to_pose(trajectoryClient, {
            "arm;to": y + 0.04, # + 0.08 for cm offset from aruco with center of bottle
        })
        
        rospy.sleep(4)
            
        move_to_pose(trajectoryClient, {
                "stretch_gripper;to": -40,
            })
        rospy.loginfo("Object grab attempted.") 
        
        rospy.sleep(3)

        move_to_pose(trajectoryClient, {
            "lift;to": z + 0.10,
        })

        # Begin STOW procedure

        move_to_pose(trajectoryClient, {
           "arm;to" : 0,
        })

        move_to_pose(trajectoryClient, {
           "wrist_yaw;to" : np.pi,
        })

        rospy.sleep(3)

        if moveUntilContact:
            self.move_until_contact(
                trajectoryClient,
                {
                    "lift;to": z - 0.2,
                }
            )
        
        move_to_pose(trajectoryClient, {
            "head_pan;to": 0,
        }
        )

        rospy.loginfo("Picking pipeline complete")
        rospy.sleep(5)


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
            if self.isContact:
                break
            move_to_pose(trajectory_client, {
                'lift;to': curz - 0.01,
            })
            rospy.sleep(0.05)
        move_to_pose(trajectory_client, { # moving up slightly so that the grasp is accurate
            'lift;to': curz + 0.04,
        })  
    
