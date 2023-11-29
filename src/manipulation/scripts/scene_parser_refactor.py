import numpy as np
import cv2
from sensor_msgs.msg import JointState, CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from yolo.msg import Detections
import rospy
import time
import random
import colorsys
import tf
from visualization_msgs.msg import Marker

from geometry_msgs.msg import Vector3, Vector3Stamped, PoseStamped, Point, Pose, PointStamped, Point32
from sensor_msgs.msg import PointCloud2
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
import open3d as o3d
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud, PointField
import sensor_msgs.point_cloud2 as pcl2
import actionlib
from yolo.msg import DeticDetectionsGoal, DeticDetectionsAction, DeticDetectionsActionResult
from manipulation.msg import DeticRequestAction, DeticRequestGoal, DeticRequestResult

class ObjectLocation:
    
    def __init__(self, time_diff_for_realtime):
        
        self.time_diff_for_realtime = time_diff_for_realtime
        self.clear()

    def clear(self):
        self.positions = []
        self.orientations = []
        self.times = []
        self.confidences = []
        self.n_items = 0
        
    def add_observation(self, x, y, z, roll, pitch, yaw, confidence, t):
        self.positions.append([x, y, z])
        self.orientations.append([roll, pitch, yaw])
        self.times.append(t)
        self.confidences.append(confidence)
        self.n_items += 1
    
    def get_estimate(self):

        if self.n_items == 0:
            return None, False
        
        weights = np.array(self.confidences)

        positions = np.array(self.positions)
        orientations = np.array(self.orientations)
        x = np.average(positions[:, 0], weights = weights)
        y = np.average(positions[:, 1], weights = weights)
        z = np.average(positions[:, 2], weights = weights)
        avg_roll = np.average(orientations[:, 0], weights = weights)
        avg_pitch = np.average(orientations[:, 1], weights = weights)
        avg_yaw = np.average(orientations[:, 2], weights = weights)
                
        angleToGo = np.arctan2(y, x) #orient to face the aruco
        radius = np.sqrt(x**2 + y**2)
        
        dic = {
            'x' : x,
            'y' : y,
            'z' : z,
            'angleToGo' : angleToGo,
            'radius' : radius,
            'roll' : avg_roll,
            'pitch' : avg_pitch,
            'yaw' : avg_yaw,
        }
        
        return dic, True

    def get_latest_observation(self):
        if self.n_items == 0:
            return None, False
        isLive = False
        latest_position = self.positions[-1]
        latest_orientation = self.orientations[-1]
        latest_time = self.times[-1]
        latest_confidence = self.confidences[-1]
        dic = {
            'x' : latest_position[0],
            'y' : latest_position[1],
            'z' : latest_position[2],
            'roll' : latest_orientation[0],
            'pitch' : latest_orientation[1],
            'yaw' : latest_orientation[2],
            'confidence' : latest_confidence,
            'time' : latest_time,
        }
        timediff = time.time() - latest_time
        # rospy.loginfo("Time diff = " + str(timediff))
        if timediff < self.time_diff_for_realtime:
            isLive = True
        return dic, isLive


class SceneParser:
    def __init__(self):

        self.bridge = CvBridge()
        self.depth_image_subscriber = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.rgb_image_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.boundingBoxSub = message_filters.Subscriber("/object_bounding_boxes", Detections)
        
        self.obj_cloud_pub = rospy.Publisher('/object_cloud', PointCloud2, queue_size=10)
        self.plane_cloud_pub = rospy.Publisher('/plane_cloud', PointCloud2, queue_size=10)
        self.scene_cloud_pub = rospy.Publisher('/scene_cloud', PointCloud2, queue_size=10)
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
        
        
        self.class_list  = rospy.get_param('/object_detection/class_list')
        self.time_diff_for_realtime = rospy.get_param('/manipulation/time_diff_for_realtime', 0.2)
        self.object_filter = rospy.get_param('/manipulation/object_filter', {'height' : 0.7, 'radius' : 2.5})
        
        rospy.loginfo(f"[{rospy.get_name()}]: Waiting for DETIC service...")
        self.detic_action = actionlib.SimpleActionClient('detic_predictions', DeticDetectionsAction)
        self.detic_action.wait_for_server()

        
        self.obtained_initial_images = False
        self.request_clear_object = False
        
        self.object = ObjectLocation(time_diff_for_realtime = self.time_diff_for_realtime)
        self.objectId = None
        
        self.current_detection = None
        self.scene_points = None
        self.object_points = None
        self.prevtime = time.time()
        
        self.tsdf_volume = None
        
        synced_messages = message_filters.ApproximateTimeSynchronizer([
            self.depth_image_subscriber, self.rgb_image_subscriber,
            self.boundingBoxSub
        ], 100, 0.5, allow_headerless=False)
        
        rospy.loginfo(f"[{rospy.get_name()}]: Waiting for camera intrinsics...")
        msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=30)
        self.intrinsics = np.array(msg.K).reshape(3,3)
        self.cameraParams = {
            'width' : msg.width,
            'height' : msg.height,
            'intrinsics' : np.array(msg.K).reshape(3,3),
            'focal_length' : self.intrinsics[0][0],
            "distortion_coefficients" : np.array(msg.D).reshape(5,1),
        }
        
        # self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        # self.arucoParams = cv2.aruco.DetectorParameters_create()
        

        
        
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.arucoParams =  cv2.aruco.DetectorParameters()
        
        cam = o3d.camera.PinholeCameraIntrinsic()
        cam.intrinsic_matrix = self.intrinsics
        cam.width = msg.width
        cam.height = msg.height
        
        self.o3d_cam = cam
        
        rospy.loginfo(f"Obtained camera intrinsics.")

        self.parse_mode = ("YOLO", None)
        

        
        self.listener = tf.TransformListener()
        self.tf_ros = tf.TransformerROS()
        self.tf_broadcaster = tf.TransformBroadcaster()
        rospy.sleep(2)
        
        synced_messages.registerCallback(self.parse_scene)
        
        starttime = time.time()
        while not self.obtained_initial_images:
            rospy.loginfo("Waiting for initial images")
            rospy.sleep(0.5)
            if time.time() - starttime > 10:
                rospy.loginfo("Failed to obtain images. Please check the camera node.")
                exit()
        
        

        rospy.loginfo("Scene parser initialized")

    

    def set_parse_mode(self, mode, id = None):
        """
        track mode = "ARUCO" or mode = "YOLO"
        id is the object to track or the id of the aruco marker
        """
        self.parse_mode = mode
        if id is not None:
            self.objectId = id
        self.clear_observations(wait = True)

    
                     
    def clear_observations(self, wait = False):
        self.request_clear_object = True
        while wait and self.request_clear_object:
            rospy.sleep(0.1)
            
    def clearAndWaitForNewObject(self, numberOfCopies = 10):
        rospy.loginfo("Clearing object location array")
        self.clear_observations(wait = True)
        rospy.loginfo("Object location array cleared. Waiting for new object location")
        startTime = time.time()
        while self.object.n_items <= numberOfCopies:
            rospy.loginfo(f"Accumulated copies = {self.object.n_items} ")
            if time.time() - startTime >= 10:
                rospy.loginfo("Timed out waiting to detect object..")
                return False
            rospy.sleep(0.5)
        return True
    
    def get_latest_observation(self):
        return self.object.get_latest_observation()
    
    def estimate_object_location(self):
        return self.object.get_estimate()
    
    def is_grasp_success(self):
        """Checks if the grasp was successful or not

        Returns:
            bool: True if successful, False otherwise
        """
        # returns true perenially for now since grasp success recognition is disabled.
        return True
    
    def parse_detections(self, detection):
        # get object detections
        num_detections = (detection.nPredictions)
        msg = {
            "boxes" : np.array(detection.box_bounding_boxes).reshape(num_detections, 4),
            "box_classes" : np.array(detection.box_classes).reshape(num_detections),
            'confidences' : np.array(detection.confidences).reshape(num_detections),
        }

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
            
            # TODO publish point cloud here instead of saving image like this.
             
            if time.time() - self.prevtime > 1:
                viz = self.color_image.copy().astype(np.float32)
                viz /= np.max(viz)
                viz *= 255
                viz = viz.astype(np.uint8)
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(viz, (int(x_f), int(y_f)), 5, (255, 0, 0), -1)
                viz = cv2.resize(viz, (0, 0), fx = 0.25, fy = 0.25)
                cv2.imwrite('/home/hello-robot/alfred-autonomy/src/manipulation/scripts/cropped.png', viz)
                self.prevtime = time.time()
            
            point = self.convertPointToFrame(x_f, y_f, z_f, "base_link")
            x, y, z = point.x, point.y, point.z
            # print(x, y, z)
            self.tf_broadcaster.sendTransform((x, y, z),
                tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time.now(),
                "object_pose",
                "base_link"
            )
            
            radius = np.sqrt(x**2 + y**2)
            confidence /= radius
            
            if (z > self.object_filter['height']  and radius < self.object_filter['radius']): # to remove objects that are not in field of view
                self.current_detection = [x1, y1, x2, y2]
                self.object.add_observation(
                    x = x,
                    y = y,
                    z = z,
                    roll = 0,
                    pitch = 0,
                    yaw = 0,
                    confidence = confidence,
                    t = time.time()
                )
                
                

    def parse_aruco(self, image):
        
        (corners, ids, _) = cv2.aruco.detectMarkers(image, 
                                                    self.dictionary,
                                                    parameters=self.arucoParams
                                                    )
        intrinsics = self.cameraParams['intrinsics']
        distorition_coefficients = self.cameraParams['distortion_coefficients']
        if ids is not None:

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix=intrinsics, distCoeffs=distorition_coefficients)
            for rvec, tvec, id in zip(rvecs, tvecs, ids):
                if id == self.objectId: # looking only for a single ID please.
      
                    # quat = cv2.Rodrigues(rvec)[0]
                    # quaternion = np.array([quat[0][0],quat[1][0],quat[2][0],1.0])
                    # quaternion = quaternion/np.linalg.norm(quaternion)
                    tf_aruco_to_cam = np.zeros((4, 4))
                    tf_aruco_to_cam[3, 3] = 1.0
                    tf_aruco_to_cam[:3, 3] = tvec[0]
                    tf_aruco_to_cam[:3, :3], _ = cv2.Rodrigues(rvec)
                    quaternion = tf.transformations.quaternion_from_matrix(tf_aruco_to_cam)
                    self.tf_broadcaster.sendTransform(
                        tvec[0], 
                        quaternion,
                        rospy.Time.now(),
                        "aruco_pose" + str(id),
                        "camera_color_optical_frame"
                    )
                    tf_cam_to_base = self.get_transform("base_link", "camera_color_optical_frame")
                    tf_cam_to_base = np.vstack((tf_cam_to_base, np.array([0, 0, 0, 1])))
                    tf_aruco_to_base = np.matmul(tf_cam_to_base, tf_aruco_to_cam)
                    
                    x, y, z = tf_aruco_to_base[:3, 3]
                    roll, pitch, yaw = tf.transformations.euler_from_matrix(tf_aruco_to_base)
                    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
                    self.object.add_observation(
                        x = x,
                        y = y,
                        z = z,
                        roll = roll,
                        pitch = pitch,
                        yaw = yaw,
                        confidence = 1.0,
                        t = time.time()
                    )
                    self.current_detection = [x - 0.1, y - 0.1, x + 0.1, y + 0.1]
                     
                    self.tf_broadcaster.sendTransform(
                        [x, y, z], 
                        quaternion,
                        rospy.Time.now(),
                        "aruco_pose_1" + str(id),
                        "base_link"
                    )
    
    def check_if_in_frustum(self, point):
        frustum_lines = self.get_view_frustum()[1]
        frustum_lines = np.array(frustum_lines)
        for line in frustum_lines:
            v0, v1 = line
            normal = np.cross(v1 - v0, point - v0)
            d = -np.dot(normal, v0)
            if np.dot(normal, point) + d < 0:
                return False
                
        return True
    
    
    def get_view_frustum(self, max_depth = 2500, publish = True):
        """
        returns the points of the frustum of the camera w.r.t base_link
        """
        
        points_cam_frame = np.array([
            [0, 0, 50],
            [self.cameraParams['width'], 0, 50],
            
            [0, self.cameraParams['height'], 50],
            [self.cameraParams['width'], self.cameraParams['height'], 50],
            
            [0, 0, max_depth],
            [self.cameraParams['width'], 0, max_depth],
            
            [0, self.cameraParams['height'], max_depth],
            [self.cameraParams['width'], self.cameraParams['height'], max_depth],
            
        ], dtype=np.float32)
        points_cam_frame[:, 2] /= 1000.0
        
        
        frustum_lines = [
            [points_cam_frame[0], points_cam_frame[1]],
            [points_cam_frame[1], points_cam_frame[3]],
            [points_cam_frame[3], points_cam_frame[2]],
            [points_cam_frame[2], points_cam_frame[0]],
            
            [points_cam_frame[4], points_cam_frame[5]],
            [points_cam_frame[5], points_cam_frame[7]],
            [points_cam_frame[7], points_cam_frame[6]],
            [points_cam_frame[6], points_cam_frame[4]],
            
            [points_cam_frame[0], points_cam_frame[4]],
            [points_cam_frame[1], points_cam_frame[5]],
            [points_cam_frame[2], points_cam_frame[6]],
            [points_cam_frame[3], points_cam_frame[7]],
        ]
        
        new_points = []
        # for point in points_cam_frame:
            # new_points.append(self.convertPointToFrame(point[0], point[1], point[2]))
        final_frustum_lines = []
        for pair in frustum_lines:
            temp = []
            for point in pair:
                conv = self.convertPointToFrame(point[0], point[1], point[2])
                new_points.append(conv)
                temp.append([conv.x, conv.y, conv.z])
            final_frustum_lines.append(temp.copy())
                
                
            
        # add lines to visualize frustum
        if publish:
            rospy.loginfo("Publishing frustum")
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"
            marker = Marker()
            marker.header = header
            marker.type = Marker.LINE_LIST
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.005  # Line width
            marker.color.r = 0
            marker.color.g = 1
            marker.color.b = 0
            marker.color.a = 1
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration(5)
            
            # marker.frame_locked = True
            marker.id = 500

            strip_points =[]
            for point in new_points:
                # print(point)
                strip_points.append(Point(x = point.x, y = point.y, z = point.z))
                
            marker.points = strip_points
            
            self.marker_pub.publish(marker)
        
        # return np.array(new_points)            
        return new_points, final_frustum_lines
        
        
        

    def parse_scene(self, depth, rgb, detection):
        self.obtained_initial_images = True

        # rospy.loginfo("Got synced images, delta = " + str(time.time() - self.prevtime))
        # self.prevtime = time.time()
        
        # self.get_view_frustum(publish = True)
        
        
        # convert images to numpy arrays
        depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        self.depth_image = np.array(depth_img, dtype=np.float32)
        
        color_img = self.bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")
        self.color_image = np.array(color_img, dtype=np.uint8)
        
        if self.request_clear_object:
            self.object.clear()
            self.request_clear_object = False
        
        
        if self.objectId is None:
            return False # no object id set
        
        if self.parse_mode == "ARUCO":
            self.parse_aruco(self.color_image)
        elif self.parse_mode == "YOLO":
            self.parse_detections(detection)
        
        return True
    
    def convert_point_from_to_frame(self, x, y, z, from_frame, to_frame):
        camera_point = PointStamped()
        camera_point.header.frame_id = from_frame
        camera_point.point.x = x
        camera_point.point.y = y
        camera_point.point.z = z
        point = self.listener.transformPoint(to_frame, camera_point).point
        return [point.x, point.y, point.z]
        
    
    def convertPointToFrame(self, x_f, y_f, z, to_frame = "base_link"):
        intrinsic = self.intrinsics
        fx = intrinsic[0][0]
        fy = intrinsic[1][1] 
        cx = intrinsic[0][2] 
        cy = intrinsic[1][2]

        x_gb = (x_f - cx) * z/ fx
        y_gb = (y_f - cy) * z/ fy

        camera_point = PointStamped()
        camera_point.header.frame_id = 'camera_color_optical_frame'
        camera_point.point.x = x_gb
        camera_point.point.y = y_gb
        camera_point.point.z = z
        point = self.listener.transformPoint(to_frame, camera_point).point
        return point

    def get_transform(self, target_frame, source_frame):
        listener = self.listener
        while not rospy.is_shutdown():
            try:
                t = listener.getLatestCommonTime(target_frame, source_frame)
                (trans,rot) = listener.lookupTransform(target_frame, source_frame, t)

                trans_mat = np.array(trans).reshape(3, 1)
                rot_mat = R.from_quat(rot).as_matrix()
                transform_mat = np.hstack((rot_mat, trans_mat))
                return transform_mat
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.loginfo("Transform not found between " + target_frame + " and " + source_frame + " at " + str(rospy.Time.now()))
                continue
    
    def reset_tsdf(self, resolution = 0.05):
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= resolution,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
    
    def get_pcd_from_tsdf(self, publish = False):
        pcd = self.tsdf_volume.extract_point_cloud()
        points = np.array(pcd.points)
        points = points[points[:, 2] < 1.6]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if publish:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"  # Set your desired frame_id
            msg = pcl2.create_cloud_xyz32(header, points)
            self.scene_cloud_pub.publish(msg)
            
        return pcd
    
    
    def get_marker(
        self, 
        id,
        frame = 'base_link', 
        center = [0, 0, 0],
        mtype = Marker.CUBE,
        extents = [0.05, 0.05, 0.05],
        color = None,
        alpha = 1,
        euler_angles = [0, 0, 0]
    ):
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = rospy.Time.now()
        marker.header.stamp = rospy.Time.now()
        
        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = mtype
        marker.id = id

        # Set the scale of the marker
        marker.scale.x = extents[0]
        marker.scale.y = extents[1]
        marker.scale.z = extents[2]

        # Set the color
        
        
        if color is None:
            # generate bright random color unique to the uid
            random.seed(id)
            h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
            color = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
            
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = alpha

        # Set the pose of the marker
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]
        quat = R.from_euler('xyz', euler_angles).as_quat()

        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        
        return marker
    
    def capture_shot(self, publish = False, use_detic = False):
        rospy.sleep(0.1)
        extrinsics = self.get_transform('camera_color_optical_frame', 'base_link')
        extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)
        object_properties, isLive = self.get_latest_observation()
        if isLive:
            x, y, z = object_properties['x'], object_properties['y'], object_properties['z']
            detection = [x, y, z]
            marker = self.get_marker(
                id = 10, 
                center = detection,
                mtype = Marker.SPHERE
            )
            self.marker_pub.publish(marker)
            
        self.get_detic_detections()

        depthimg = self.depth_image.copy()
        rgbimg = self.color_image.copy()  
        depth = o3d.geometry.Image(depthimg)
        color = o3d.geometry.Image(rgbimg.astype(np.uint8))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000, convert_rgb_to_intensity=False
        )

        self.tsdf_volume.integrate(
                rgbd_image,
                self.o3d_cam,
                extrinsics)
    
    def get_detic_detections(self):
        req = DeticDetectionsGoal(
            image = self.bridge.cv2_to_imgmsg(self.color_image, encoding="passthrough"),
        )
        rospy.loginfo("Requested DETIC for results")
        starttime = time.time()
        self.detic_action.send_goal(req)
        wait = self.detic_action.wait_for_result()
        detection : DeticDetectionsActionResult = self.detic_action.get_result()
        rospy.loginfo("Got from DETIC in {} seconds".format(time.time() - starttime))
        
        num_detections = detection.nPredictions
        msg = {
            "boxes" : np.array(detection.box_bounding_boxes).reshape(num_detections, 4),
            "box_classes" : np.array(detection.box_classes).reshape(num_detections),
            'confidences' : np.array(detection.confidences).reshape(num_detections),
            "radii" : [],
            'base_link_point' : []
        }
        
        final_msg = {
            "boxes" : [],
            "box_classes" : [],
            'confidences' : [],
            "radii" : [],
            'base_link_point' : []
        }
        
        for i in range(num_detections):
            box = np.squeeze(msg['boxes'][i]).astype(int)
            confidence = np.squeeze(msg['confidences'][i])
            x1, y1, x2, y2 =  box
            crop = self.depth_image[y1 : y2, x1 : x2]
            
            instance_id = i + 1
            
            mask = self.bridge.imgmsg_to_cv2(detection.seg_mask, desired_encoding="passthrough")
            mask = np.array(mask).copy()
            
            mask[mask != instance_id] = 0
            mask[mask == instance_id] = 1
            z_f = np.median(crop[crop != 0])/1000.0

            x_f = (x1 + x2)/2
            y_f = (y1 + y2)/2
            
            point = self.convertPointToFrame(x_f, y_f, z_f, "base_link")
            x, y, z = point.x, point.y, point.z

            radius = np.sqrt(x**2 + y**2)
            
            msg['base_link_point'].append([x, y, z])
            msg['radii'].append(radius)
            if (z > self.object_filter['height']  and radius < self.object_filter['radius']): # to remove objects that are not in field of view
                final_msg['boxes'].append([x1, y1, x2, y2])
                final_msg['box_classes'].append(msg['box_classes'][i])
                final_msg['confidences'].append(msg['confidences'][i])
                final_msg['radii'].append(msg['radii'][i])
                final_msg['base_link_point'].append(msg['base_link_point'][i])
        # print(msg['box_classes'])
        # print(np.where(np.array(msg['box_classes']).astype(np.uint8) == self.objectId))
        loc = np.where(np.array(msg['box_classes']).astype(np.uint8) == self.objectId)[0]
        if len(loc) > 0:
            loc = loc[0]
            box = np.squeeze(msg['boxes'][loc]).astype(int)
            confidence = np.squeeze(msg['confidences'][loc])
            x1, y1, x2, y2 =  box
            crop = self.depth_image[y1 : y2, x1 : x2]
            
            instance_id = loc + 1
            
            mask = self.bridge.imgmsg_to_cv2(detection.seg_mask, desired_encoding="passthrough")
            mask = np.array(mask).copy()
            
            mask[mask != instance_id] = 0
            mask[mask == instance_id] = 1
            
            z_f = np.median(crop[crop != 0])/1000.0
            x_f = (x1 + x2)/2
            y_f = (y1 + y2)/2

            
            viz = self.color_image.copy().astype(np.float32)
            viz /= np.max(viz)
            viz *= 255
            viz = viz.astype(np.uint8)
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(viz, (int(x_f), int(y_f)), 5, (255, 0, 0), -1)
            viz[:, :, 0][mask != 0] = 255
            viz = cv2.resize(viz, (0, 0), fx = 0.25, fy = 0.25)
            cv2.imwrite('/home/hello-robot/alfred-autonomy/src/manipulation/scripts/detic.png', viz)
            self.prevtime = time.time()
        
            point = self.convertPointToFrame(x_f, y_f, z_f, "base_link")
            x, y, z = point.x, point.y, point.z

            self.tf_broadcaster.sendTransform(msg['base_link_point'][loc],
                tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time.now(),
                "object_pose",
                "base_link"
            )
            radius = np.sqrt(x**2 + y**2)
            confidence /= radius
            
            if (z > self.object_filter['height']  and radius < self.object_filter['radius']): # to remove objects that are not in field of view
                return [x1, y1, x2, y2], mask, final_msg, True
                
        return None, None, final_msg, False
    
    def get_point_cloud_from_image(self, depth_image, rgb_image, workspace_mask):
        """
        workspace mask is a binary mask of the same size as depth image
        """
        rospy.loginfo("Generating point cloud from image...")
        starttime = time.time()
        
        dimg = depth_image.copy()
        dimg *= workspace_mask
        dimg = dimg.astype(np.float32)
        
        colors = rgb_image.copy()
        colors = colors.astype(np.float32)
        colors /= 255.0
        
        xs = np.arange(0, self.cameraParams['width']).astype(np.float32)
        ys = np.arange(0, self.cameraParams['height']).astype(np.float32)
        
        xx, yy = np.meshgrid(xs, ys)
        xx = xx.flatten()
        yy = yy.flatten()
        zz = dimg.flatten()
        colors = colors.reshape(-1, 3)        

        filt_indices = zz != 0
        xx = xx[filt_indices]
        yy = yy[filt_indices]
        zz = zz[filt_indices]
        colors = colors[filt_indices]
        
        
        cx = self.intrinsics[0][2]
        cy = self.intrinsics[1][2]
        fx = self.intrinsics[0][0]
        fy = self.intrinsics[1][1]
        
        zz = zz / 1000.0
        xx = (xx - cx) * zz / fx
        yy = (yy - cy) * zz / fy
        
        points_cam_frame = np.vstack((xx, yy, zz, np.ones_like(xx))).T
        
        transform = self.get_transform("base_link", "camera_color_optical_frame")

        points_base_frame = np.matmul(transform, points_cam_frame.T).T
        points_cam_frame = points_cam_frame[:, :3]
        points_base_frame = points_base_frame[:, :3]
        
        rospy.loginfo("Generated point cloud. Took {} seconds.".format(time.time() - starttime))
        
        return points_base_frame, colors
        
    def get_plane(self, height_filter = 0.5, dist_filter = 1, publish = False):
        points = self.scene_points
        if points is None:
            return None
        
        plane_bounds = []
        plane_height = np.nan
        possible_plane_points = points[points[:, 2] > height_filter]
        radii = np.linalg.norm(possible_plane_points[:, :2], axis = 1)
        possible_plane_points = possible_plane_points[radii < dist_filter]
        
        
        if possible_plane_points.shape[0] < 1000:
            rospy.loginfo("Too few points to fit plane. Aborting.")
            return None
        
        
        rospy.loginfo("Fitting plane....")
        starttime = time.time()
        
        o3dpcd = o3d.geometry.PointCloud()
        o3dpcd.points = o3d.utility.Vector3dVector(possible_plane_points)
        plane_model, inliers = o3dpcd.segment_plane(distance_threshold=0.02,
                                                        ransac_n=5,
                                                        num_iterations=2000)
        inlier_points = np.array(o3dpcd.select_by_index(inliers).points)
        
        plane_bounds = [
            np.min(inlier_points[:, 0]), 
            np.max(inlier_points[:, 0]),
            np.min(inlier_points[:, 1]),
            np.max(inlier_points[:, 1]),
        ]
        plane_height = np.mean(inlier_points[:, 2])
        
        rospy.loginfo("Generated plane. Took {} seconds.".format(time.time() - starttime))
        
        if publish:
            rospy.loginfo("Publishing plane cloud")
            starttime = time.time()
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"  # Set your desired frame_id
            msg = pcl2.create_cloud_xyz32(header, inlier_points)
            self.plane_cloud_pub.publish(msg)
            rospy.loginfo("Published plane cloud. Took" + str(time.time() - starttime) + " seconds.")
            
        return plane_bounds, plane_height
    
    def get_placing_location(self, plane, height_of_object, publish = False):
        rospy.loginfo("Generating point to palce")
        plane_bounds, plane_height = plane
        xmin, xmax, ymin, ymax = plane_bounds
        
        if xmin * xmax < 0: # this means that it is safe to place without moving base
            placingLocation = np.array([
                0.0, 
                (ymin + ymax)/2, 
                plane_height + height_of_object + 0.1
            ])
        placingLocation = np.array([
            (xmin + xmax)/2, 
            (ymin + ymax)/2,
            plane_height + height_of_object + 0.1
        ])
        rospy.loginfo("Placing point generated...")
        if publish:
            rospy.loginfo("Publishing placing location")
            self.tf_broadcaster.sendTransform((placingLocation),
                tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time.now(),
                "placing_location",
                "base_link"
            )
            rospy.loginfo("Published transform for placing location")
            
        return placingLocation
    
    def get_point_cloud_o3d(self, ws_mask = None, downsample = None,publisher = None):
        """
        sets everything as numpy array
        """
        starttime = time.time()
        rospy.loginfo("Generating point cloud...")
        
        depth = self.depth_image.copy()
        color_image = self.color_image.copy() 
        
        depth[depth < 150] = 0
        depth[depth > 3000] = 0
        
        if ws_mask is None:
            ws_mask = np.ones_like(depth)
            
        depth *= ws_mask
        
        depth = o3d.geometry.Image(depth)
        color = o3d.geometry.Image(color_image.astype(np.uint8))
        
        extrinsics = self.get_transform('camera_color_optical_frame', 'base_link')
        extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, convert_rgb_to_intensity=False
        )
        cam = o3d.camera.PinholeCameraIntrinsic()
        cam.intrinsic_matrix = self.intrinsics
        cam.width = color_image.shape[1]
        cam.height = color_image.shape[0]
        
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, cam, extrinsics
        )
        
        if downsample is not None:
            rospy.loginfo("Subsampling cloud")
            cloud = cloud.voxel_down_sample(voxel_size=downsample)
        
        points = np.array(cloud.points)
        rospy.loginfo("Point cloud generated in {} seconds.".format(time.time() - starttime))
        
        if publisher is not None:
            rospy.loginfo("Publishing point cloud")
            starttime = time.time()
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"  # Set your desired frame_id
            msg = pcl2.create_cloud_xyz32(header, points)
            publisher.publish(msg)
            rospy.loginfo("Published point cloud. Took" + str(time.time() - starttime) + "seconds.")
            
        return points
    
    def publish_point(self, point, name = "point", frame = "base_link"):
        self.tf_broadcaster.sendTransform((point[0], point[1], point[2]),
            tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            name,
            frame
        )
    
    def set_point_cloud(self, visualize = False, use_detic = False, publish_object = False, publish_scene =False, get_object = True):
        rospy.loginfo(f"[{rospy.get_name()}]: Setting point cloud")
        
        if get_object:
            success = False
            if use_detic:           
                result, seg_mask, _, success = self.get_detic_detections()
            
            if not success:
                self.object_points = None
                self.scene_points = None
                rospy.loginfo("No object found")
                return
                # if self.current_detection:
                #     x1, y1, x2, y2 =  np.array(self.current_detection).astype(int)
                # else:
                #     x1 = 0
                #     y1 = 0
                #     x2 = self.cameraParams['width']
                #     y2 = self.cameraParams['height']
                # self.ws_mask = np.zeros_like(self.depth_image)
                # self.ws_mask[y1 : y2, x1 : x2] = 1
            else:
                x1, y1, x2, y2 =  result
                self.ws_mask = seg_mask 
                rospy.loginfo("Object found")
        else:
            self.ws_mask = np.zeros_like(self.depth_image)
            

        object_publisher = None
        scene_publisher = None
        if publish_object:
            object_publisher = self.obj_cloud_pub
        if publish_scene:
            scene_publisher = self.scene_cloud_pub


        self.object_points = self.get_point_cloud_o3d(ws_mask = self.ws_mask, publisher = object_publisher)

        no_object_mask = np.ones_like(self.depth_image) - self.ws_mask
        self.scene_points = self.get_point_cloud_o3d(ws_mask = no_object_mask, publisher = scene_publisher)

        if visualize:
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(self.object_points)
            
            o3d.visualization.draw_geometries([scene_pcd])
        
    
    def get_grasp(self, visualize = False, publish_grasp = False, publish_cloud = False):
        rospy.loginfo("Generating grasp")
        starttime = time.time()
        if self.object_points is None:
            rospy.loginfo("No object cloud found, please set it first.")
            return None
        
        points = self.object_points
        # segment out the object and remove things that are not part of the object using depth information
        o3dpcd = o3d.geometry.PointCloud()
        o3dpcd.points = o3d.utility.Vector3dVector(points)

        rospy.loginfo("Clustering object cloud")        
        labels = np.array(o3dpcd.cluster_dbscan(eps=0.01, min_points=50, print_progress=False))
        max_label = labels.max()
        rospy.loginfo("Found {} clusters".format(max_label + 1))

        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0
        
        labels += 1
        
        cluster_sizes = np.bincount(labels)
        largest_cluster = np.argmax(cluster_sizes[1:]) + 1
        o3dpcd.points = o3d.utility.Vector3dVector(points[labels == largest_cluster])
        # o3dpcd.colors = o3d.utility.Vector3dVector(self.object_colors[labels == largest_cluster])
        points = np.array(o3dpcd.points)
        
        ori_bbox = o3dpcd.get_oriented_bounding_box()
        # axi_bbox = o3dpcd.get_axis_aligned_bounding_box()
        
        box_angles = np.array(ori_bbox.R)
        extents = np.array(ori_bbox.extent)
        
        long_axis = np.argmax(extents)
        unit_vecs_bbox = box_angles.T
        axis_vec = unit_vecs_bbox[long_axis]
        
        rospy.loginfo("Extents of the bounding box are {}".format(extents))
        
        grasp_center = np.array(ori_bbox.center)
        grasp_yaw = np.arctan2(axis_vec[1], axis_vec[0])

        if grasp_yaw < 0:
            grasp_yaw += np.pi/2
        else:
            grasp_yaw -= np.pi/2    
        rospy.loginfo("Generated grasp. Took {} seconds.".format(time.time() - starttime))
        
        if publish_grasp:
            rospy.loginfo("Publishing grasp pose")
            self.tf_broadcaster.sendTransform((grasp_center),
                tf.transformations.quaternion_from_euler(0, 0, grasp_yaw),
                rospy.Time.now(),
                "grasp_pose",
                "base_link"
            )
            box_marker = self.get_marker(
                id = 30,
                center = grasp_center,
                mtype = Marker.CUBE,
                extents = extents,
                euler_angles = R.from_matrix(box_angles).as_euler('xyz'),
                frame = "base_link",
                alpha = 0.3
            )

            self.marker_pub.publish(box_marker)
        
        if publish_cloud:
            rospy.loginfo("Publishing point cloud")
            starttime = time.time()
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"  # Set your desired frame_id
            msg = pcl2.create_cloud_xyz32(header, points)
            self.obj_cloud_pub.publish(msg)
            rospy.loginfo("Published point cloud. Took" + str(time.time() - starttime) + "seconds.")
        
        if visualize:
            line_set = o3d.geometry.LineSet()
            h, w, l = extents
            lineLength = max(h, w, l) / 2
            axis_ind = long_axis 
            line_set.points = o3d.utility.Vector3dVector(np.array(
                [
                    grasp_center, 
                    grasp_center - unit_vecs_bbox[axis_ind] * lineLength,
                    
                    grasp_center + unit_vecs_bbox[axis_ind - 2] * w/2 - unit_vecs_bbox[axis_ind] * lineLength,
                    grasp_center + unit_vecs_bbox[axis_ind - 2] * w/2 + unit_vecs_bbox[axis_ind] * lineLength,
                    
                    grasp_center - unit_vecs_bbox[axis_ind - 2] * w/2 - unit_vecs_bbox[axis_ind] * lineLength,
                    grasp_center - unit_vecs_bbox[axis_ind - 2] * w/2+ unit_vecs_bbox[axis_ind] * lineLength,
                    
                    grasp_center + unit_vecs_bbox[axis_ind - 1] * w/2 - unit_vecs_bbox[axis_ind] * lineLength,
                    grasp_center + unit_vecs_bbox[axis_ind - 1] * w/2 + unit_vecs_bbox[axis_ind] * lineLength,
                    
                    grasp_center - unit_vecs_bbox[axis_ind - 1] * w/2 - unit_vecs_bbox[axis_ind] * lineLength,
                    grasp_center - unit_vecs_bbox[axis_ind - 1] * w/2+ unit_vecs_bbox[axis_ind] * lineLength, 

                    
                ])
            )
            
            
            theta = np.arccos(axis_vec @ np.array([1, 0, 0]))
            rmat = o3d.geometry.get_rotation_matrix_from_xyz((0, theta, 0))
            
            lines = [
                [0, 1], 
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
            ]
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0, 1, 0])
            
            grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius = 0.01)
            grasp_sphere.compute_vertex_normals()
            grasp_sphere.paint_uniform_color([1, 0, 0])
            grasp_sphere.translate(grasp_center)
                    

            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(self.scene_points)
            scene_pcd.colors = o3d.utility.Vector3dVector(self.scene_colors)

            
            o3d.visualization.draw_geometries([o3dpcd, 
                                            grasp_sphere,
                                            scene_pcd,
                                            #    axis_aligned_bbox,
                                                line_set,
                                                ori_bbox,
                                            ])
        
        return [grasp_center, grasp_yaw]
    
if __name__ == "__main__":
    
    rospy.init_node("scene_parser")
    import json
    parser = SceneParser()
    # parser.set_parse_mode("YOLO", 9)
    
    for i in range(100):
        parser.get_detic_detections()
        rospy.sleep(1)
    rospy.spin()
    # idx = str(12).zfill(6)
    
    # rgb_image = cv2.imread(f'/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/images/rgb/{idx}.png')
    # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    # depth_image = cv2.imread(f'/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/images/depth/{idx}.tif', cv2.IMREAD_UNCHANGED).astype(np.uint16)
    
    # with open(f'/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/images/pose/{idx}.json', 'r') as f:
    #     data = json.load(f)
        
    # intrinsics = data['intrinsics']
    # transform = np.array(data['transform'])
    # transform = np.vstack((transform, np.array([0, 0, 0, 1])))
    
    # parser.depth_image = depth_image
    # parser.color_image = rgb_image
    # parser.transform = transform 
    # parser.transform = np.linalg.inv(transform)
    # parser.intrinsics = np.array(intrinsics).reshape(3, 3)
    
    # h, w = depth_image.shape[:2]
    # print(h, w)
    # parser.cameraParams = {
    #     'width' : w,
    #     'height' : h,
    #     'intrinsics' : np.array(intrinsics).reshape(3,3),
    #     'focal_length' : intrinsics[0][0],
    # }
  
    # num_detections = 1
    # parser.current_detection = {
    #     "boxes" : np.array([
    #         # [350,200, 660, 280]     # 12 - bottle
    #         [531, 341, 625, 409] # 12 - crumple
    #     ]),
    #     "box_classes" : [8],
    #     'confidences' : [1.0],
    # }
    
    # parser.set_object_id(8)
    # parser.set_point_cloud()
    # parser.get_grasp()
    
    
    
    # rospy.spin()