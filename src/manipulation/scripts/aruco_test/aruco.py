#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import CameraInfo
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import time
import tf
# from visual import AlignToObject

def get_camera_params(camera_info_topic):
    """
    Retrieves camera matrix and distortion coefficients from a camera info topic.

    Args:
        camera_info_topic (str): Name of the camera info topic.

    Returns:
        tuple: Camera matrix and distortion coefficients as numpy arrays.
    """
    camera_matrix = None
    distortion_coefficients = None

    def camera_info_callback(msg):
        print("Inside camera info callback")
        nonlocal camera_matrix, distortion_coefficients
        camera_matrix = np.array(msg.K).reshape((3, 3))
        distortion_coefficients = np.array(msg.D)

    # Subscribe to the camera info topic and wait for first message
    rospy.loginfo(f"Subscribing to camera info topic '{camera_info_topic}'...")
    sub = rospy.Subscriber(camera_info_topic, CameraInfo, camera_info_callback)
    print("About to see camera matrix")
    while camera_matrix is None or distortion_coefficients is None:
        rospy.sleep(0.1)
        print("Seeing camera matrix")
    print("Saw camera matrix")
    sub.unregister()
    rospy.loginfo("Received camera parameters:")
    rospy.loginfo(f"Camera matrix: {camera_matrix}")
    rospy.loginfo(f"Distortion coefficients: {distortion_coefficients}")

    return camera_matrix, distortion_coefficients

class ArUcoDetector:

    def __init__(self):

        print("Inside the Aruco Detector library")

        # Load ArUco dictionary
        # self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()

        # Initialize camera parameters
        self.camera_matrix, self.distortion_coefficients = get_camera_params("/camera/color/camera_info")

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.pose_pub = rospy.Publisher("/aruco/pose", PoseStamped, queue_size=1)
        self.aruco_tf_broadcaster = tf.TransformBroadcaster()
        self.image_pub = rospy.Publisher("/aruco/image", Image, queue_size=1)
        self.detections = []
        self.lastDetectedtime = time.time()
        self.ids = None

    def image_callback(self, msg):

        # Convert ROS image message to OpenCV image
        detections = []
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # print('ros time:',time.time())
        # cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)

        # Detect ArUco markers
        (corners, ids, rejected) = cv2.aruco.detectMarkers(cv_image, self.dictionary,
        parameters=self.arucoParams)

        # Estimate poses of detected markers
        if ids is not None:
            # print(ids)
        # Draw the detected markers and their IDs on the color image
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

            # Estimate the pose of each marker in the world coordinate system
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.042, cameraMatrix=self.camera_matrix, distCoeffs=self.distortion_coefficients)
            # Publish poses of detected markers
            for rvec, tvec, id in zip(rvecs, tvecs, ids):
                pose_msg = PoseStamped()
                pose_msg.header = msg.header
                pose_msg.pose.position.x = tvec[0][0]
                pose_msg.pose.position.y = tvec[0][1]
                pose_msg.pose.position.z = tvec[0][2]
                quat = cv2.Rodrigues(rvec)[0]
                pose_msg.pose.orientation.x = quat[0][0]
                pose_msg.pose.orientation.y = quat[1][0]
                pose_msg.pose.orientation.z = quat[2][0]
                pose_msg.pose.orientation.w = 1.0
                detections.append((id, pose_msg))

                quaternion = np.array([quat[0][0],quat[1][0],quat[2][0],1.0])
                quaternion = quaternion/np.linalg.norm(quaternion)
                self.aruco_tf_broadcaster.sendTransform((tvec[0][0], tvec[0][1], tvec[0][2]),quaternion,rospy.Time.now(),"aruco_pose","camera_color_optical_frame")
                
                if(id==0):
                    # print("Detected Aruco ID: ", id)
                    self.pose_pub.publish(pose_msg)
            self.lastDetectedTime = time.time()
            self.detections = detections
            self.ids = ids
        else:
            self.detections = []

if __name__ == '__main__':
    rospy.init_node('arucodetector')
    aruco_detector = ArUcoDetector()
    print("Inside the main function")
    rospy.spin()