#!/usr/bin/env python3

import stretch_body.robot as rb
import numpy as np
import time
import cv2
import pyrealsense2 as rs
import rospy
from aruco import ArUcoDetector
from helpers import move_to_pose
# from hri import HRI
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped

from control_msgs.msg import FollowJointTrajectoryAction
from manip_basic_control import ManipulationMethods
import actionlib
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_srvs.srv import Trigger, TriggerResponse
import tf

class MoveRobot:

    def __init__(self):
        # rospy.init_node('move_robot')

        # self.hri = HRI(self.deliverIceCream)
        self.x=0
        self.y=0
        self.z=0
        self.arucos = ArUcoDetector()
        rospy.loginfo("waiting for aruco detector")
        self.trajectoryClient = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo("waiting for trajectory client")
        self.trajectoryClient.wait_for_server()
        self.manipulationMethods = ManipulationMethods()
        self.listener : tf.TransformListener = tf.TransformListener()
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

        startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
        startManipService()
        rospy.loginfo("waiting for stow robot service")
        self.stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
        self.stow_robot_service.wait_for_service()
        self.stow_robot_service()

        rospy.sleep(3)
        rospy.loginfo("Every service is running\n")
        
    def getCommand(self):
        return True
    
    def getTransform(self):
        print("Inside the transform function")
        from_frame_rel = '/camera_color_optical_frame'
        to_frame_rel = '/base_link'
        lastCommonTime =  self.listener.getLatestCommonTime(to_frame_rel, from_frame_rel)
        while not rospy.is_shutdown():
            try:
                trans, rot = self.listener.lookupTransform(to_frame_rel, from_frame_rel, lastCommonTime)
                trans_inv = [-x for x in trans]
                rot_mat_inv = R.from_quat(rot).inv().as_matrix()

                trans_mat = np.array(trans).reshape(3, 1)
                rot_mat = R.from_quat(rot).as_matrix()
                transform_mat = np.hstack((rot_mat, trans_mat))
                print("Sending transform from camera frame to base frame")
                return transform_mat
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            
        rospy.logerr(f"[{rospy.get_name()}] " + "Transform not found between {} and {}".format(to_frame_rel, from_frame_rel))
        return None
    

    def publishMarker(self, x, y, z):
        marker = Marker()

        marker.header.frame_id = "base_link"  # Set the frame ID
        # marker.header.stamp = rospy.Time.now()
        # marker.ns = "your_namespace"
        # marker.id = 0
        marker.type = Marker.SPHERE  # Set the marker type to a sphere
        marker.action = Marker.ADD

        # Set the scale of the sphere
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Set the color of the sphere (RGBA values)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the position of the sphere
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        # Publish the marker
        self.marker_pub.publish(marker)

    def getArucoLocation(self, index = 0) -> PoseStamped:
        print("Inside the function")
        detections = self.arucos.detections
        ids = [det[0] for det in detections]
        print("detections received")
        # while index not in ids:
        #     detections = self.arucos.detections
        #     ids = [det[0] for det in detections]
        #     continue
        print("going fine")
        try:
            
            idx = ids.index(index)
            _,  detection = detections[idx]
            print("goingfine")
            x, y, z= detection.pose.position.x, detection.pose.position.y, detection.pose.position.z
            print("Retreiving transforms")
            transform = self.getTransform()
            # self.publishTransform(transform, 'base_link', 'your_visualized_frame')

            print("\nBottle coordinates before transformation:",x,y,z)
            # x, y, z = np.dot(translation, np.array([1, 0, 0])), np.dot(translation, np.array([0, 1, 0])), np.dot(translation, np.array([0, 0, 1]))
            aruco_coord_in_base_frame = np.dot(transform, np.array([[x, y, z, 1]]).T)
            print("ARUCO", aruco_coord_in_base_frame)
            self.publishMarker(x, y, z)
            
            print("\nBottle coordinates after transformation:",x,y,z)
            # rospy.sleep(30)
            # print(x, y, z)


            return _, [x, y, z]
        except:
            print("Aruco not found")

    def deliverIceCream(self):
        # self.moveToGraspPose()
        
        # pick random detection
        
        # print(detections)
        # detection = np.random.choice(detections)
        det, _ = self.getArucoLocation()
        # self.rotateBase()
        # # self.moveToGraspPose()
        # det, _ = self.getArucoLocation()
        # self.alignWithAruco()
        _,[x, y, z], _ = self.getArucoLocation()
        self.grasp([0, y, z])
        # self.moveToTable()
        # self.place()
        rospy.loginfo("Pick cycle completed")

    # def alignWithAruco(self):
    #     rospy.loginfo("Aligning object horizontally")
    #     det ,  trans = self.getArucoLocation()
    #     move_to_pose(self.trajectoryClient, {
    #         'base_translate;by' : trans[0],
    #     }) 
    #     rospy.sleep(5)
        # prevxerr = np.inf
        # xerr = np.inf
        # curvel = 0.0
        # kp = 0.7
        # kd = 0.1
        # prevsettime = time.time()
        # while abs(xerr) > 0.008:
        #     det ,  trans = self.getArucoLocation()
        #     xerr = trans[0]
        #     rospy.loginfo("Alignment Error = " + str(xerr))
        #     dxerr = (xerr - prevxerr)
        #     vx = (kp * xerr + kd * dxerr)
        #     prevxerr = xerr
        #     move_to_pose(self.trajectoryClient, {
        #             'base_translate;vel' : vx,
        #     }) 
        #     curvel = vx
        #     prevsettime = time.time()
            
        #     rospy.sleep(0.1)
        # move_to_pose(self.trajectoryClient, {
        #     'base_translate;vel' : 0.0,
        # }) 
        # # risk
        # # if self.isDetected is False:
        # #     return False
        # return True
    # def rotateBase(self):
    #     rospy.loginfo(f"Moving robot towards object by {abs(self.y-0.4)} meters")
    #     if(self.y>0.4):    
    #         move_to_pose(self.trajectoryClient, {
    #             'base_translate;by': (abs(self.y)-0.4),
    #         })
    #     rospy.sleep(5)
    #     move_to_pose(self.trajectoryClient, {
    #         'base_translate;vel' : 0.4,
    #     })
    #     rospy.sleep(1)
    #     # move_to_pose(self.trajectoryClient, {
    #     #     'base_translate;vel' : 0.0,
    #     # })
    #     rospy.sleep(6)
    #     move_to_pose(self.trajectoryClient, {
    #         'base_rotate;by': np.pi/2,
    #     })
    #     rospy.sleep(3)
    #     move_to_pose(self.trajectoryClient, {
    #         "head_pan;to" : -np.pi/2})
    #     rospy.sleep(5)

    def run(self):
        move_to_pose(
            self.trajectoryClient,
            {
                'head_tilt;to': -20 * np.pi/180,
            }
        )
        rospy.sleep(3)
        self.deliverIceCream()
        # while not rospy.is_shutdown():
            # # rospy.sleep(3)
            # # fetch_ice_cream = self.hri.run()
            # fetch_ice_cream = True
            # if fetch_ice_cream:
            #     rospy.loginfo("fetching ice cream...")
            #     self.deliverIceCream()
            #     break

    def place(self):
        self.manipulationMethods.place(self.trajectoryClient, 0.5, 0.5, 1, 0)
        self.stow_robot_service()
        move_to_pose(self.trajectoryClient, {
            'base_rotate;by' : -np.pi/2
        })

    def grasp(self, a):
        # pass
        x, y, z= a[0], a[1], a[2]
        # transform = self.getTransform()
        # x, y, z = np.dot(translation, np.array([1, 0, 0])), np.dot(translation, np.array([0, 1, 0])), np.dot(translation, np.array([0, 0, 1]))
        # x, y, z = np.dot(transform, np.array([x, y, z, 1]))[:3]
        # print(x, y, z)
        self.x=x
        self.y=y
        self.z=z
        self.manipulationMethods.move_to_pregrasp(self.trajectoryClient) #got at height, opens the gripper and makes yaw 0
        distToExtend = abs(abs(y) - self.manipulationMethods.getEndEffectorPose()[0]) #+ self.offset_dict[self.label2name[goal.objectId]]
        self.manipulationMethods.pick(
            self.trajectoryClient, 
            x, 
            # distToExtend, 
            y,
            # z + 0.05,
            z, 
            0,
            moveUntilContact = False
        ) 
        self.stow_robot_service()

        

    # def moveToGraspPose(self):
    #     move_to_pose(self.trajectoryClient, {
    #         'base_rotate;by': np.pi/2,
    #     })
    #     rospy.sleep(2)
    #     move_to_pose(self.trajectoryClient, {
    #         'base_translate;by': 0.5,
    #     })
    #     rospy.sleep(5)
    #     move_to_pose(self.trajectoryClient, {
    #         'base_rotate;by': np.pi/2,
    #     })
    #     rospy.sleep(0.5)
    #     move_to_pose(self.trajectoryClient, {
    #         "head_pan;to" : -np.pi/2
    #     })
    #     rospy.sleep(0.5)
    #     move_to_pose(self.trajectoryClient, {
    #         # 'head_pan;to': -np.pi/2,
    #         'head_tilt;to': -20  * np.pi/ 180,
    #     })
    #     rospy.sleep(2)



    # def moveToTable(self):
    #     # pass
    #     move_to_pose(
    #         self.trajectoryClient,
    #         {
    #             "base_rotate;by" : np.pi/2
    #         }
    #     )
    #     rospy.sleep(3)
    #     move_to_pose(
    #         self.trajectoryClient,
    #         {
    #             "base_translate;by" : 2
    #         }
    #     )
    #     rospy.sleep(10)
    #     move_to_pose(
    #         self.trajectoryClient,
    #         {
    #             "base_rotate;by" : np.pi/2
    #         }
    #     )
    #     rospy.sleep(5)
    #     # move_to_pose(
    #     #     self.trajectoryClient,
    #     #     {
    #     #         "base_rotate;by" : -np.pi/2
    #     #     }
    #     # )

    

if __name__ == "__main__":
    rospy.init_node('move_robot')
    moveRobot = MoveRobot()
    moveRobot.run()