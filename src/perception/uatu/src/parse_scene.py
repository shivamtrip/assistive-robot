#!/usr/bin/env python3
import rospy
import time
import numpy as np
from sensor_msgs.msg import JointState, CameraInfo, Image
from enum import Enum
import tf
import tf.transformations
from tf import TransformerROS
from cv_bridge import CvBridge, CvBridgeError
import cv2
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger
from geometry_msgs.msg import Vector3, Vector3Stamped, PoseStamped, Point, Pose, PointStamped
# ROS action finite state machine
from yolo.msg import Detections
from graph import Graph, Node

class SceneParser:
    def __init__(self,):
        self.node_name = 'uatu'
        rospy.init_node(self.node_name, anonymous=True)

        self.listener = tf.TransformListener()
        self.tf_ros = TransformerROS()
        
        rospy.loginfo(f"[{rospy.get_name()}]: Waiting for camera intrinsics...")
        msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=10)
        self.cameraParams = {
            'width' : msg.width,
            'height' : msg.height,
            'intrinsics' : np.array(msg.K).reshape(3,3),
            'focal_length' : self.intrinsics[0][0],
        }
        self.intrinsics = np.array(msg.K).reshape(3,3)
        rospy.loginfo(f"[{rospy.get_name()}]: Obtained camera intrinsics.")

        self.bridge = CvBridge()
        
        self.obtainedInitialImages = False

        self.objectLocArr = []
        self.upscale_fac = 1/rospy.get_param('/image_shrink/downscale_ratio')
        
        self.depth_image_subscriber = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image,self.depthCallback)
        self.boundingBoxSub = rospy.Subscriber("/object_bounding_boxes", Detections, self.parseScene)
        
        self.class_list = rospy.get_param('class_list')
        self.graph : Graph = Graph(len(self.class_list))
        
        self.trackedNode = Node(-1)
        
        while not self.obtainedInitialImages:
            rospy.sleep(0.1)
        
        
        rospy.Service('/clear_scene', Trigger, self.clear_graph)
        # TODO add custom service to start tracking object
        rospy.Service('/start_tracking_object', Trigger, self.start_tracking_object)
        rospy.Service('/stop_tracking_object', Trigger, self.stop_tracking_object)
        
        
        rospy.loginfo(f"[{self.node_name}]: Initial images obtained.")
        rospy.loginfo(f"[{self.node_name}]: Node initialized")
        
        
    
    
    def start_tracking_object(self, req):
        self.tracking = req.objectId
        
    def stop_tracking_object(self, req):
        self.tracking = -1
        self.trackedNode.clear_samples()
        
    def clear_graph(self, req):
        self.graph.clear()
        rospy.loginfo(f"[{self.node_name}]: Graph cleared.")

    def convertPointToBaseFrame(self, x_f, y_f, z):
        intrinsic = self.intrinsics
        fx = intrinsic[0][0]
        fy = intrinsic[1][1] 
        cx = intrinsic[0][2] 
        cy = intrinsic[1][2]

        x_gb = (x_f - cx) * z/ fx
        y_gb = (y_f - cy) * z/ fy

        camera_point = PointStamped()
        camera_point.header.frame_id = '/camera_depth_optical_frame'
        camera_point.point.x = x_gb
        camera_point.point.y = y_gb
        camera_point.point.z = z
        point = self.listener.transformPoint('base_link', camera_point).point
        return point

    
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
    


    def parseScene(self, msg : Detections):
        self.obtainedInitialImages = True
        num_detections = (msg.nPredictions)
        msg = {
            "boxes" : np.array(msg.box_bounding_boxes).reshape(num_detections, 4),
            "box_classes" : np.array(msg.box_classes).reshape(num_detections),
            'confidences' : np.array(msg.confidences).reshape(num_detections),
        }
        n = len(msg['box_classes'])
        
        upscale_fac = self.upscale_fac
        for i in range(n):
            box = np.squeeze(msg['boxes'][n])
            confidence = np.squeeze(msg['confidences'][n])
            class_type = np.squeeze(msg['box_classes'][n])
            x1, y1, x2, y2 =  box
            crop = self.depth_image[int(y1 * upscale_fac) : int(y2 * upscale_fac), int(x1 * upscale_fac) : int(x2 * upscale_fac)]
            
            z_f = np.median(crop[crop != 0])/1000.0
            h, w = self.depth_image.shape[:2]
            y_new1 = w - x2 * upscale_fac
            x_new1 = y1 * upscale_fac
            y_new2 = w - x1 * upscale_fac
            x_new2 = y2 * upscale_fac

            x_f = (x_new1 + x_new2)/2
            y_f = (y_new1 + y_new2)/2
            
            # instead of center, randomly sample points within the bounding box and obtain the mean
            n_points = 30
            x_s = np.random.randint(x_new1, x_new2, n_points)
            y_s = np.random.randint(y_new1, y_new2, n_points)
            z_s = np.array([self.depth_image[y, x] for x, y in zip(x_s, y_s)])/1000.0
            
            points = np.array([self.convertPointToBaseFrame(x, y, z) for x, y, z in zip(x_s, y_s, z_s)])
            x, y, z = np.median(points, axis=0)

            radius = np.sqrt(x**2 + y**2)
            confidence /= radius
            if (radius < 2.5): # to remove objects that are not in field of view
                self.graph.add_observation(
                    x, y, z,
                    class_type, confidence, 
                    msg.header.stamp)
                if self.tracking == class_type:
                    self.trackedNode.add_sample(x, y, z, None, msg.header.stamp, confidence)

    def depthCallback(self, depth_img):
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        depth_img = np.array(depth_img, dtype=np.float32)
        self.depth_image = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
    

if __name__ == "__main__":
    node = SceneParser()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass