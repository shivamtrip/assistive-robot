#!/usr/bin/env python3

from control_msgs.msg import FollowJointTrajectoryGoal
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from helpers import move_to_pose
from std_msgs.msg import String, Header
import time
import json
import numpy as np
from sensor_msgs.msg import JointState, CameraInfo, Image
from enum import Enum
# from helpers import armclient, headclient, diffdrive, gripperclient
import tf
import tf.transformations
from tf import TransformerROS
from tf import TransformListener, TransformBroadcaster
from cv_bridge import CvBridge, CvBridgeError
from helpers import convertPointToDepth
import cv2
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2, PointField
import pandas as pd
# ROS action finite state machine
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryResult
import actionlib
from yolo.msg import Detections
from scene_parser import SceneParser

class State(Enum):
    SEARCH = 1
    
    MOVE_TO_OBJECT = 3
    ROTATE_90 = 4
    ALIGN_HORIZONTAL = 5
    COMPLETE = 6

    # substates
    SCANNING = 11      
    ALIGN_TO_OBJECT = 12          

    FAILED = -1


class AlignToObject:
    def __init__(self, scene_parser : SceneParser):
        self.node_name = 'visual_alignment'
        self.state = State.SEARCH
        
        self.scene_parser = scene_parser
        self.pid_gains = rospy.get_param('/manipulation/pid_gains')
        
        self.prevxerr = 0    
        
        self.recoveries = rospy.get_param('/manipulation/recoveries')
        self.mean_err_horiz_alignment = rospy.get_param('/manipulation/mean_err_horiz_alignment')
        self.vs_range = rospy.get_param('/manipulation/vs_range') 
        self.vs_range[0] *= np.pi/180
        self.vs_range[1] *= np.pi/180
        
        self.moving_avg_n = rospy.get_param('/manipulation/moving_average_n')
        self.max_graspable_distance = rospy.get_param('/manipulation/max_graspable_distance')
        
        self.head_tilt_angle_search = rospy.get_param('/manipulation/head_tilt_angle_search')
        self.head_tilt_angle_grasp = rospy.get_param('/manipulation/head_tilt_angle_grasp')
        
        
        self.forward_vel = rospy.get_param('/manipulation/forward_velocity')
        self.interval_before_restart = rospy.get_param('/manipulation/forward_velocity')
        
        self.obj_tf_publisher = tf.TransformBroadcaster()
        
        rospy.loginfo(f"[{self.node_name}]: Waiting for trajectory server")
        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.trajectoryClient.wait_for_server()
        
        
        rospy.loginfo(f"[{self.node_name}]: Initial images obtained.")
        rospy.loginfo(f"[{self.node_name}]: Node initialized")
        self.prevtime = time.time()
        
    
    def findAndOrientToObject(self):
        
        rospy.loginfo(f"[{self.node_name}]: Finding and orienting to object")
        
        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : - self.head_tilt_angle_search * np.pi/180,
            'head_pan;to' : self.vs_range[0],
        })
        
        nRotates = 0
        while not rospy.is_shutdown():
            for i, angle in enumerate(np.linspace(self.vs_range[0], self.vs_range[1], self.vs_range[2])):
                move_to_pose(self.trajectoryClient, {
                    'head_pan;to' : angle,
                })
                rospy.sleep(self.vs_range[3]) #settling time.

            [angleToGo, x, y, z, radius], success = self.scene_parser.estimate_object_location()
            self.scene_parser.clear_observations()
            if success:
                self.objectLocation = [x, y, z]
                rospy.loginfo('Object found at angle: {}. Location = {}. Radius = {}'.format(angleToGo, (x, y, z), radius))
                move_to_pose(self.trajectoryClient, {
                    'base_rotate;by' : angleToGo,
                    'head_pan;to' : 0,
                })
                return True
            else:
                rospy.loginfo("Object not found, rotating base.")
                self.scene_parser.clear_observations()
                move_to_pose(self.trajectoryClient, {
                    'base_rotate;by' : -np.pi/2,
                    'head_pan;to' : self.vs_range[0],
                })
                rospy.sleep(5)
                nRotates += 1
                
            if nRotates >= self.recoveries['n_scan_attempts']:
                return False
        
        return False

    def recoverFromFailure(self):
        rospy.loginfo("Attempting to recover from failure")        
        move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : -self.forward_vel,
        })
        rospy.sleep(1)
        move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : 0.0,
        })


    def moveTowardsObject(self):
        rospy.loginfo("Moving towards the object")
        lastDetectedTime = time.time()
        startTime = time.time()
        vel = self.forward_vel
        intervalBeforeRestart = self.interval_before_restart

        move_to_pose(self.trajectoryClient, {
            'head_tilt;to' : - self.head_tilt_angle_grasp * np.pi/180,
        })
       
        if not self.scene_parser.clearAndWaitForNewObject():
            return False

        maxGraspableDistance = self.max_graspable_distance
        result, success = self.scene_parser.estimate_object_location()
        if success:
            distanceToMove = result[3] - maxGraspableDistance
        else:
            return False
        rospy.loginfo("distanceToMove = {}".format(distanceToMove))
        if distanceToMove < 0:
            rospy.loginfo("Object is close enough. Not moving.")
            return True
        
        while not rospy.is_shutdown():
            motionTime = time.time() - startTime
            distanceMoved = motionTime * vel
            
            move_to_pose(self.trajectoryClient, {
                'base_translate;vel' : vel,
            })
            rospy.loginfo("Distance to move = {}, Distance moved = {}".format(distanceToMove, distanceMoved))

            lastDetectedTime = time.time()
            object_location, isLive = self.scene_parser.get_latest_observation()
            if isLive:
                if object_location[0] < maxGraspableDistance:
                    rospy.loginfo("Object is close enough. Stopping.")
                    rospy.loginfo("Object location = {}".format(self.scene_parser.objectLocArr[-1]))
                    move_to_pose(self.trajectoryClient, {
                        'base_translate;vel' : 0.0,
                    })          
                    return True
            if distanceMoved >= distanceToMove - 0.3:
                if time.time() - lastDetectedTime > intervalBeforeRestart:
                    rospy.loginfo("Lost track of the object. Going back to search again.")
                    move_to_pose(self.trajectoryClient, {
                        'base_translate;vel' : 0.0,
                    })  
                    return False

            rospy.sleep(0.1)

        
        return False

    def alignObjectForManipulation(self):
        """
        Rotates base to align manipulator for grasping
        """
        rospy.loginfo(f"[{self.node_name}]: Aligning object for manipulation")
        move_to_pose(self.trajectoryClient, {
                'head_pan;to' : -np.pi/2,
                'base_rotate;by' : np.pi/2,
            }
        )
        rospy.sleep(4)
        return True
     
    def alignObjectHorizontal(self, ee_pose_x = 0.0, debug_print = {}):
        """
        Visual servoing to align gripper with the object of interest.
        """
        
        self.requestClearObject = True
        if not self.scene_parser.clearAndWaitForNewObject(2):
            return False

        xerr = np.inf
        rospy.loginfo("Aligning object horizontally")
        
        prevxerr = np.inf
        moving_avg_n = self.moving_avg_n
        moving_avg_err = []
        mean_err = np.inf
        
        kp = self.pid_gains['kp']
        kd = self.pid_gains['kd']
        curvel = 0.0
        prevsettime = time.time()
        while len(moving_avg_err) <= moving_avg_n and  mean_err > self.mean_err_horiz_alignment:
            [x, y, z, conf, pred_time], isLive = self.scene_parser.get_latest_observation()
            # if isLive:
            rospy.loginfo("Time since last detection = {}".format(time.time() - pred_time))
            xerr = (x - ee_pose_x)
            
                
            moving_avg_err.append(xerr)
            if len(moving_avg_err) > moving_avg_n:
                moving_avg_err.pop(0)
                
            dxerr = (xerr - prevxerr)
            vx = (kp * xerr + kd * dxerr)
            prevxerr = xerr
            
            mean_err = np.mean(np.abs(np.array(moving_avg_err)))
            debug_print["xerr"] = xerr
            debug_print["mean_err"] = mean_err
            debug_print["x, y, z"] = [x, y, z]
            
            print(debug_print)

            move_to_pose(self.trajectoryClient, {
                    'base_translate;vel' : vx,
            }, asynchronous= True) 
            curvel = vx
            prevsettime = time.time()
            # else:
            #     rospy.logwarn("Object not detected. Using open loop motions")
            #     xerr = (x - curvel * (time.time() - prevsettime)) - ee_pose_x
            
            
            rospy.sleep(0.1)
        
        move_to_pose(self.trajectoryClient, {
            'base_translate;vel' : 0.0,
        }, asynchronous= True) 
        
        return True

    def main(self):
        rospy.loginfo("Triggered Visual Servoing")
        
        while not rospy.is_shutdown():
            if self.state == State.SEARCH:
                success = self.findAndOrientToObject()                
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                self.state = State.MOVE_TO_OBJECT

            elif self.state == State.MOVE_TO_OBJECT:
                success = self.moveTowardsObject()
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                self.state = State.ROTATE_90

            elif self.state == State.ROTATE_90:
                success = self.alignObjectForManipulation()
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                self.state = State.COMPLETE

            elif self.state == State.ALIGN_HORIZONTAL:
                success = self.alignObjectHorizontal()
                if not success:
                    rospy.loginfo("Object not found. Exiting visual servoing")
                    self.reset()
                    return False
                self.state = State.COMPLETE

            elif self.state == State.COMPLETE:
                break
            rospy.sleep(0.5)
        rospy.loginfo("Visual servoing is complete!")
        self.reset()
        return True
    
        
    def reset(self):
        self.state = State.SEARCH
        
    

if __name__ == "__main__":
    # unit test
    switch_to_manipulation = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    switch_to_manipulation.wait_for_service()
    switch_to_manipulation()
    rospy.init_node("align_to_object", anonymous=True)
    node = AlignToObject(8)
    # node.main(8)
    node.findAndOrientToObject()
    # node.moveTowardsObject()
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass