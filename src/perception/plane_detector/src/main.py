#!/usr/local/lib/obj_det_env/bin/python3

from pathlib import Path
import rospy
from sensor_msgs.msg import PointCloud2
# import ros_numpy
import numpy as np
# import open3d as o3d
import matplotlib.pyplot as plt
from fit_plane import fit_plane
from error_func import ransac_error
from plot_result import plot_plane
import rospy
import tf
import actionlib
from scipy.spatial.transform import Rotation as R
from plane_detector.msg import PlaneDetectAction, PlaneDetectResult
import time


class PlaneDetector:

    latest_point_cloud = None

    def __init__(self):

        rospy.init_node("plane_detector_node")
        
        self.server = actionlib.SimpleActionServer('plane_detector', PlaneDetectAction, self.getPlane, False)

        self.point_cloud_topic = rospy.get_param('plane_detector/point_cloud_topic')
        self.base_frame = rospy.get_param('plane_detector/base_frame')
        self.camera_frame = rospy.get_param('plane_detector/camera_frame')
        self.listener = tf.TransformListener()

        rospy.loginfo("Started plane detector node")

        # RANSAC parameters:
        #TODO(@shivamt): tune params.
        self.confidence = rospy.get_param('plane_detector/confidence')
        self.inlier_threshold = rospy.get_param('plane_detector/inlier_threshold')
        self.min_sample_distance = rospy.get_param('plane_detector/min_sample_distance')


        # RANSAC parameters:
        #TODO(@shivamt): tune params.
        self.confidence = rospy.get_param('plane_detector/confidence')
        self.inlier_threshold = rospy.get_param('plane_detector/inlier_threshold')
        self.min_sample_distance = rospy.get_param('plane_detector/min_sample_distance')

        self.server.start()
    # TODO(@shivamt) : add feedbacks

    def get_transform(self):

        while not rospy.is_shutdown():
            try:
                t = self.listener.getLatestCommonTime(self.base_frame, self.camera_frame)
                (trans,rot) = self.listener.lookupTransform(self.base_frame, self.camera_frame, t)

                trans_mat = np.array(trans).reshape(3, 1)
                rot_mat = R.from_quat(rot).as_matrix()
                transform_mat = np.hstack((rot_mat, trans_mat))

                return transform_mat
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue


    def sendResult(self, place_point, best_plane, num_iterations):
        # TODO(@shivamt) : add feedbacks
        plane_detect_action = PlaneDetectResult()

        plane_detect_action.x = place_point[0]
        plane_detect_action.y = place_point[1]
        plane_detect_action.z = place_point[2]
        plane_detect_action.a = best_plane[0]
        plane_detect_action.b = best_plane[1]
        plane_detect_action.c = best_plane[2]
        plane_detect_action.d = best_plane[3]
        plane_detect_action.success = True

        if num_iterations == 250 or place_point[2] < 0.4:      
            plane_detect_action.success = False

        self.server.set_succeeded(plane_detect_action)

    def getPlane(self, goal):

        rospy.loginfo("Waiting to receive a point cloud")
        self.start_cloud_subscriber()

        while not self.latest_point_cloud:
            continue

        rospy.loginfo("Point cloud received")
        self.sub.unregister()

        # Converting point cloud to numpy array without ros_numpy

        pcd_points = np.frombuffer(self.latest_point_cloud.data, dtype=np.float32).reshape(-1, self.latest_point_cloud.width, 5)[:, :, :3][0]
        
        # pcd_points=np.zeros((pc.shape[0],3))
        # pcd_points[:,0]=pc['x']
        # pcd_points[:,1]=pc['y']
        # pcd_points[:,2]=pc['z']
        

        pcd_points = np.hstack((pcd_points,np.ones((pcd_points.shape[0],1)))) 
        transform_mat = self.get_transform()            
        pcd_points = np.matmul(transform_mat,pcd_points.T).T  


        indices = abs(pcd_points[:,  1]) < 1.5           
        indices = np.logical_and(indices ,(abs(pcd_points[:,  2]) < 0.85))
        pcd_new = pcd_points[indices]               
        
        # Apply plane-fitting algorithm
        best_plane, best_inliers, num_iterations = fit_plane(pcd_new, confidence=self.confidence,
                                                            inlier_threshold=self.inlier_threshold,
                                                            min_sample_distance=self.min_sample_distance,
                                                            error_func=ransac_error)

        # Filter inlier points from pointcloud
        pcd_inliers = pcd_new[best_inliers]                    
    
        # Transform filter points from camera frame to base frame
            

        # Get median of transformed points (assumed to be approx. plane center)    
        place_point = np.median(pcd_inliers,axis=0)  

        # Send result to action client           
        self.sendResult(place_point, best_plane, num_iterations)
        rospy.loginfo("Sent result to client")

        # Visualize plane in Open3D
        # plot_plane(pcd_inliers, place_point)                


    def storePointcloud(self, data):
        self.latest_point_cloud = data


    def start_cloud_subscriber(self):
        self.sub = rospy.Subscriber(self.point_cloud_topic, PointCloud2, self.storePointcloud)
        

if __name__ == '__main__':

    node = PlaneDetector()

    rospy.spin()