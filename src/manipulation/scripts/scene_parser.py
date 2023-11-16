import numpy as np
import cv2
from sensor_msgs.msg import JointState, CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from yolo.msg import Detections
import rospy
import time
import tf
from geometry_msgs.msg import Vector3, Vector3Stamped, PoseStamped, Point, Pose, PointStamped
from scipy.spatial import transform as R

class SceneParser:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth_image_subscriber = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.rgb_image_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.boundingBoxSub = message_filters.Subscriber("/object_bounding_boxes", Detections)
        
        synced_messages = message_filters.ApproximateTimeSynchronizer([self.depth_image_subscriber, self.rgb_image_subscriber, self.boundingBoxSub], 60, 0.5, allow_headerless=False)
        synced_messages.registerCallback(self.parse_scene)
        
        self.obtained_initial_images = False
        
        self.request_clear_object = False
        
        self.listener = tf.TransformListener()
        self.tf_ros = tf.TransformerROS()
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        
        self.time_diff_for_realtime = rospy.get_param('/manipulation/time_diff_for_realtime', 0.2)
        self.object_filter = rospy.get_param('/manipulation/object_filter', {'height' : 0.7, 'radius' : 2.5})

        rospy.loginfo(f"[{rospy.get_name()}]: Waiting for camera intrinsics...")
        msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=10)
        self.intrinsics = np.array(msg.K).reshape(3,3)
        self.cameraParams = {
            'width' : msg.width,
            'height' : msg.height,
            'intrinsics' : np.array(msg.K).reshape(3,3),
            'focal_length' : self.intrinsics[0][0],
        }
        rospy.loginfo(f"[{rospy.get_name()}]: Obtained camera intrinsics.")

        
        self.objectLocArr = []
        self.objectId = None
        
        rospy.loginfo("Waiting for initial images")
        while not self.obtained_initial_images:
            rospy.sleep(0.5)
            
            
    def set_object_id(self, object_id):
        self.objectId = object_id
            
    def clear_observations(self):
        self.request_clear_object = True
    
    def clearAndWaitForNewObject(self, numberOfCopies = 10):
        rospy.loginfo("Clearing object location array")
        while self.request_clear_object:
            rospy.sleep(0.1)
            
        rospy.loginfo("Object location array cleared. Waiting for new object location")
        startTime = time.time()
        while len(self.objectLocArr) <= numberOfCopies:
            rospy.loginfo(f"Accumulated copies = {len(self.objectLocArr)} ")
            if time.time() - startTime >= 10:
                rospy.loginfo("Object not found :(")
                return False
            rospy.sleep(0.5)

        return True
    
    def get_latest_observation(self):
        object_properties = self.objectLocArr[-1]
        x, y, z, confidence, pred_time = object_properties
        isLive = False
        
        if time.time() - pred_time < self.time_diff_for_realtime:
            isLive = True
        
        return self.objectLocArr[-1], isLive

    
    def parse_scene(self, rgb, depth, detection):
        self.obtained_initial_images = True
        
        # convert images to numpy arrays
        depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        self.depth_image = np.array(depth_img, dtype=np.float32)
        
        color_img = self.bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")
        self.color_image = np.array(color_img, dtype=np.uint8)
        
        
        # get object detections
        num_detections = (detection.nPredictions)
        msg = {
            "boxes" : np.array(detection.box_bounding_boxes).reshape(num_detections, 4),
            "box_classes" : np.array(detection.box_classes).reshape(num_detections),
            'confidences' : np.array(detection.confidences).reshape(num_detections),
        }
        
        # to ensure thread safe behavior, writing happens only in this function on a separate thread
        if self.request_clear_object:
            self.request_clear_object = False
            self.objectLocArr = []
        
        
        if self.objectId is None:
            return False # no object id set


        loc = np.where(np.array(msg['box_classes']).astype(np.uint8) == self.objectId)[0]
        
        
        if len(loc) > 0:
            loc = loc[0]
            box = np.squeeze(msg['boxes'][loc]).astype(int)
            confidence = np.squeeze(msg['confidences'][loc])
            x1, y1, x2, y2 =  box
            crop = self.depth_image[y1 : y2, x1 : x2]
            
            self.ws_mask = np.zeros_like(self.depth_image)
            self.ws_mask[y1 : y2, x1 : x2] = 1
            
            z_f = np.median(crop[crop != 0])/1000.0
            
            x_f = (x1 + x2)/2
            y_f = (y1 + y2)/2
            z_f = self.depth_image[int(y_f), int(x_f)]/1000.0
            
            
            
           
            # TODO publish point cloud here instead of saving image like this.
             
            # if time.time() - self.prevtime > 1:
            #     viz = self.color_image.copy().astype(np.float32)
            #     viz /= np.max(viz)
            #     viz *= 255
            #     viz = viz.astype(np.uint8)
            #     cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     cv2.circle(viz, (int(x_f), int(y_f)), 5, (255, 0, 0), -1)
            #     viz = cv2.resize(viz, (0, 0), fx = 0.25, fy = 0.25)
            #     cv2.imwrite('/home/hello-robot/alfred-autonomy/src/manipulation/scripts/cropped.png', viz)
            #     self.prevtime = time.time()

            point = self.convertPointToFrame(x_f, y_f, z_f, "base_link")
            x, y, z = point.x, point.y, point.z
            
            self.tf_broadcaster.sendTransform((x, y, z),
                tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time.now(),
                "object_pose",
                "base_link"
            )
            
            
            radius = np.sqrt(x**2 + y**2)
            confidence /= radius
            
            if (z > self.object_filter['height']  and radius < self.object_filter['radius']): # to remove objects that are not in field of view
                self.objectLocArr.append((x, y, z, confidence, time.time()))
    
        return True
            
    def convertPointToFrame(self, x_f, y_f, z, to_frame = "base_link"):
        intrinsic = self.intrinsics
        fx = intrinsic[0][0]
        fy = intrinsic[1][1] 
        cx = intrinsic[0][2] 
        cy = intrinsic[1][2]

        x_gb = (x_f - cx) * z/ fx
        y_gb = (y_f - cy) * z/ fy

        camera_point = PointStamped()
        camera_point.header.frame_id = '/camera_color_optical_frame'
        camera_point.point.x = x_gb
        camera_point.point.y = y_gb
        camera_point.point.z = z
        point = self.listener.transformPoint(to_frame, camera_point).point
        return point

    def get_transform(self, base_frame, camera_frame):
        listener = self.listener
        while not rospy.is_shutdown():
            try:
                t = listener.getLatestCommonTime(base_frame, camera_frame)
                (trans,rot) = listener.lookupTransform(base_frame, camera_frame, t)

                trans_mat = np.array(trans).reshape(3, 1)
                rot_mat = R.from_quat(rot).as_matrix()
                transform_mat = np.hstack((rot_mat, trans_mat))
                return transform_mat
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("Not found transform sorry.")
                continue
    
    
    def estimate_object_location(self):
        objectLocs = np.array(self.objectLocArr)
        if len(objectLocs) == 0:
            return [np.inf, np.inf, np.inf, np.inf, np.inf], False

        weights = objectLocs[:, 3]
        x = np.average(objectLocs[:, 0], weights = weights)
        y = np.average(objectLocs[:, 1], weights = weights)
        z = np.average(objectLocs[:, 2], weights = weights)
        angleToGo = np.arctan2(y, x)
        radius = np.sqrt(x**2 + y**2)
        
        return (angleToGo, x, y, z, radius), True
    
    
    def get_grasp(self):
        pass
    
    def get_plane(self):
        bounds = None
        plane_height = None
        return bounds, plane_height
    
    