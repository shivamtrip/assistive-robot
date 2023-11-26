import numpy as np
import cv2
from sensor_msgs.msg import JointState, CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from yolo.msg import Detections
import rospy
import time
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
from yolo.srv import DeticDetections, DeticDetectionsResponse, DeticDetectionsRequest


from yolo.msg import DeticDetections, DeticDetectionsResponse

class SceneParser:
    def __init__(self):

        self.bridge = CvBridge()
        self.depth_image_subscriber = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.rgb_image_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.boundingBoxSub = message_filters.Subscriber("/object_bounding_boxes", Detections)
        
        self.obj_cloud_pub = rospy.Publisher('/object_cloud', PointCloud2, queue_size=10)
        self.plane_cloud_pub = rospy.Publisher('/plane_cloud', PointCloud2, queue_size=10)
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
        
        
        rospy.loginfo(f"[{rospy.get_name()}]: Waiting for DETIC service...")
        self.detic_service = rospy.ServiceProxy('/detic_predictions', DeticDetections)
        self.detic_service.wait_for_service()

        
        self.obtained_initial_images = False
        self.request_clear_object = False
        
        self.objectLocArr = []
        self.shot_object_loc_array = []
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
        synced_messages.registerCallback(self.parse_scene)
        
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
        
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams =  cv2.aruco.DetectorParameters()
        
        self.is_moving = False
        
        cam = o3d.camera.PinholeCameraIntrinsic()
        cam.intrinsic_matrix = self.intrinsics
        cam.width = msg.width
        cam.height = msg.height
        
        self.o3d_cam = cam
        
        rospy.loginfo(f"[{rospy.get_name()}]: Obtained camera intrinsics.")

        self.parse_mode = ("YOLO", None)
        rospy.loginfo("Waiting for DETIC to load.")
        self.detic_service = rospy.ServiceProxy('detic_predictions', DeticDetections)
        self.detic_service.wait_for_service()
        
        starttime = time.time()
        while not self.obtained_initial_images:
            rospy.loginfo("Waiting for initial images")
            rospy.sleep(0.5)
            if time.time() - starttime > 10:
                rospy.loginfo("Failed to obtain images. Please check the camera node.")
                exit()
        
        self.class_list  = rospy.get_param('/object_detection/class_list')
        self.time_diff_for_realtime = rospy.get_param('/manipulation/time_diff_for_realtime', 0.2)
        self.object_filter = rospy.get_param('/manipulation/object_filter', {'height' : 0.7, 'radius' : 2.5})
        
        self.listener = tf.TransformListener()
        self.tf_ros = tf.TransformerROS()
        self.tf_broadcaster = tf.TransformBroadcaster()

        rospy.loginfo("Scene parser initialized")

    def set_parse_mode(self, mode, id = None):
        """
        track mode = "ARUCO" or mode = "YOLO"
        id is the object to track or the id of the aruco marker
        """
        self.parse_mode = mode
        self.objectId = id
        self.clear_observations(wait = True)

    
                     
    def clear_observations(self, wait = False):
        self.request_clear_object = True
        while wait and self.request_clear_object:
            rospy.sleep(0.1)
            
    def clearAndWaitForNewObject(self, numberOfCopies = 10):
        rospy.loginfo("Clearing object location array")
        self.request_clear_object = True
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
        if len(self.objectLocArr) == 0:
            return [np.inf, np.inf, np.inf, np.inf, np.inf], False
        object_properties = self.objectLocArr[-1]
        x, y, z, r, p, yaw, confidence, pred_time = object_properties
        isLive = False
        
        if time.time() - pred_time < self.time_diff_for_realtime:
            isLive = True
        
        return object_properties, isLive
    
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
            # z_f = self.depth_image[int(y_f), int(x_f)]/1000.0
            
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
            # print(x, y, z)
            # self.tf_broadcaster.sendTransform((x, y, z),
            #     tf.transformations.quaternion_from_euler(0, 0, 0),
            #     rospy.Time.now(),
            #     "object_pose",
            #     "base_link"
            # )
            radius = np.sqrt(x**2 + y**2)
            confidence /= radius
            
            if (z > self.object_filter['height']  and radius < self.object_filter['radius']): # to remove objects that are not in field of view
                # print(x, y, z, x_f, y_f, z_f)
                self.objectLocArr.append((x, y, z, 0, 0, 0, confidence, time.time()))
                self.current_detection = [x1, y1, x2, y2]

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
                    if not self.is_moving:
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
                        
                        self.objectLocArr.append((x, y, z, roll, pitch, yaw, 1.0, time.time()))
                        
                        self.tf_broadcaster.sendTransform(
                            [x, y, z], 
                            quaternion,
                            rospy.Time.now(),
                            "aruco_pose_1" + str(id),
                            "base_link"
                        )
                        
                        self.current_detection = [x - 0.1, y - 0.1, x + 0.1, y + 0.1]
                    break
                    

    def parse_scene(self, depth, rgb , detection):
        self.obtained_initial_images = True
        # rospy.loginfo("Got synced images, delta = " + str(time.time() - self.prevtime))
        
        # self.prevtime = time.time()
        
        # convert images to numpy arrays
        depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        self.depth_image = np.array(depth_img, dtype=np.float32)
        
        color_img = self.bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")
        self.color_image = np.array(color_img, dtype=np.uint8)
        
        if self.request_clear_object:
            self.request_clear_object = False
            self.objectLocArr = []
            self.shot_object_loc_array = []
        
        
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
                print("Not found transform sorry.")
                continue
    
    
    def estimate_object_location(self, from_live = True):
        objectLocs = np.array(self.objectLocArr)
        if not from_live:
            objectLocs = np.array(self.shot_object_loc_array)
        
        if len(objectLocs) == 0:
            return [np.inf, np.inf, np.inf, np.inf, np.inf], False

        weights = objectLocs[:, 6]
        # x, y, z, r, p, y, conf, time
        x = np.average(objectLocs[:, 0], weights = weights)
        y = np.average(objectLocs[:, 1], weights = weights)
        z = np.average(objectLocs[:, 2], weights = weights)
        avg_r = np.average(objectLocs[:, 3], weights = weights)
        avg_p = np.average(objectLocs[:, 4], weights = weights)
        avg_y = np.average(objectLocs[:, 5], weights = weights)
        if self.parse_mode ==  "ARUCO":
            angleToGo = np.median(objectLocs[:, 5]) - np.pi/2   

        elif self.parse_mode == "YOLO":
            angleToGo = np.arctan2(y, x) #orient to face the object
        radius = np.sqrt(x**2 + y**2)
        print("Estimated object location", x, y, z, np.rad2deg(angleToGo), radius)
        return (angleToGo, x, y, z, radius), True
    
    
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
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_link"  # Set your desired frame_id
        msg = pcl2.create_cloud_xyz32(header, points)
        self.obj_cloud_pub.publish(msg)
        return pcd
    
    def publish_point(self, point, name = "point", frame = "base_link"):
        self.tf_broadcaster.sendTransform((point[0], point[1], point[2]),
            tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            name,
            frame
        )
        
    def capture_detection(self, publish = False):
        if len(self.objectLocArr) == 0:
            return
        self.shot_object_loc_array.append(self.objectLocArr[-1])
        x, y, z, r, p, yaw, confidence, pred_time = self.objectLocArr[-1]
        if time.time() - pred_time > self.time_diff_for_realtime:
            return 
        print("Captured detection", x, y, z, r, p, yaw)
        print("Captured", self.objectLocArr[-1])
        quat = tf.transformations.quaternion_from_euler(r, p, yaw)
        self.tf_broadcaster.sendTransform(
            [x, y, z], 
            quat,
            rospy.Time.now(),
            "aruco_pose_stable",
            "base_link"
        )
    
    def capture_shot(self, publish = False):
        
        rospy.sleep(0.1)
        extrinsics = self.get_transform('camera_color_optical_frame', 'base_link')
        extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)
        object_properties, isLive = self.get_latest_observation()
        if isLive:
            x, y, z, confidence, _ = object_properties
            detection = [x, y, z]
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.header.stamp = rospy.Time.now()

            # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
            marker.type = 2
            marker.id = 10

            # Set the scale of the marker
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            # Set the color
            color=  [1, 0, 0]
            center = detection
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1

            # Set the pose of the marker
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = center[2]
            
            quat = R.from_matrix(np.eye(3)).as_quat()
            
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            self.marker_pub.publish(marker)

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
        
    
    def get_point_cloud_from_image(self, depth_image, rgb_image, workspace_mask):
        """
        workspace mask is a binary mask of the same size as depth image
        """
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
        
        # if transform is None:
        #     transform = self.transform
        # else:
        transform = self.get_transform("base_link", "camera_color_optical_frame")

        points_base_frame = np.matmul(transform, points_cam_frame.T).T
        points_cam_frame = points_cam_frame[:, :3]
        points_base_frame = points_base_frame[:, :3]
        
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
            return None
        
        o3dpcd = o3d.geometry.PointCloud()
        o3dpcd.points = o3d.utility.Vector3dVector(possible_plane_points)
        plane_model, inliers = o3dpcd.segment_plane(distance_threshold=0.02,
                                                        ransac_n=5,
                                                        num_iterations=2000)
        inlier_points = np.array(o3dpcd.select_by_index(inliers).points)
        
        if publish:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"  # Set your desired frame_id
            msg = pcl2.create_cloud_xyz32(header, inlier_points)
            self.plane_cloud_pub.publish(msg)
            
        plane_bounds = [
            np.min(inlier_points[:, 0]), 
            np.max(inlier_points[:, 0]),
            np.min(inlier_points[:, 1]),
            np.max(inlier_points[:, 1]),
        ]
        plane_height = np.mean(inlier_points[:, 2])
            
        return plane_bounds, plane_height
    
    def get_placing_location(self, plane, height_of_object, publish = False):
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
        rospy.loginfo("Placing point generated)")
        if publish:
            rospy.loginfo("Publishing placing location")
            self.tf_broadcaster.sendTransform((placingLocation),
                tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time.now(),
                "placing_location",
                "base_link"
            )
        return placingLocation
    
    def get_point_cloud_for_planning(self, publish = False):
        depth = self.depth_image.copy()
        color_image = self.color_image.copy() 
        
        depth[depth < 150] = 0
        depth[depth > 3000] = 0
        
        depth = o3d.geometry.Image(depth)
        color = o3d.geometry.Image(color_image.astype(np.uint8))
        
        extrinsics = self.get_transform('camera_color_optical_frame', 'base_link')
        extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000, convert_rgb_to_intensity=False
        )
        cam = o3d.camera.PinholeCameraIntrinsic()
        cam.intrinsic_matrix = self.intrinsics
        cam.width = color_image.shape[1]
        cam.height = color_image.shape[0]
        
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, cam, extrinsics
        )
        
        cloud = cloud.voxel_down_sample(voxel_size=0.05)
        points = np.array(cloud.points)
        self.scene_o3d_cloud = cloud

        if publish:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"  # Set your desired frame_id
            msg = pcl2.create_cloud_xyz32(header, points)
            self.obj_cloud_pub.publish(msg)
            
    def get_detic_detections(self):
        req = DeticDetectionsRequest(
            image = self.bridge.cv2_to_imgmsg(self.color_image, encoding="passthrough"),
        )
        rospy.loginfo("Requested detic")
        detection : DeticDetectionsResponse = self.detic_service(req)
        rospy.loginfo("Got from detic")
        num_detections = detection.nPredictions
        msg = {
            "boxes" : np.array(detection.box_bounding_boxes).reshape(num_detections, 4),
            "box_classes" : np.array(detection.box_classes).reshape(num_detections),
            'confidences' : np.array(detection.confidences).reshape(num_detections),
            # 'instances' : np.array(detection.instance_id).reshape(num_detections),
        }

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
                # print(x, y, z, x_f, y_f, z_f)
                # self.objectLocArr.append((x, y, z, 0, 0, 0, confidence, time.time()))
                
                return [x1, y1, x2, y2], mask, True
                
        return [np.inf, np.inf, np.inf, np.inf], None, False
        
        
        
        
    
    def set_point_cloud(self, visualize = False, publish = False, use_detic = False):
        rospy.loginfo(f"[{rospy.get_name()}]: Generating point cloud")
        
        depthimg = self.depth_image.copy()
        rgbimg = self.color_image.copy()    
        # loc = np.where(np.array(detection['box_classes']).astype(np.uint8) == self.objectId)[0]
        # box = np.squeeze(detection['boxes'][loc]).astype(int)
        # print(box)
        success = False
        if use_detic:           
            result, seg_mask, success = self.get_detic_detections()

        if not success:
            if self.current_detection:
                x1, y1, x2, y2 =  np.array(self.current_detection).astype(int)
            else:
                x1 = 0
                y1 = 0
                x2 = self.cameraParams['width']
                y2 = self.cameraParams['height']
            self.ws_mask = np.zeros_like(self.depth_image)
            self.ws_mask[y1 : y2, x1 : x2] = 1
        else:
            x1, y1, x2, y2 =  result
            self.ws_mask = seg_mask
        
            
        self.object_points, self.object_colors = self.get_point_cloud_from_image(depthimg, rgbimg, self.ws_mask)
        self.scene_points, self.scene_colors = self.get_point_cloud_from_image(depthimg, rgbimg, np.ones_like(self.depth_image))
        
        rospy.loginfo(f"[{rospy.get_name()}]: Point cloud generated.")
        
        if publish:
            rospy.loginfo("Publishing cloud")
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"  # Set your desired frame_id
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                # rgb
                PointField('r', 12, PointField.UINT8, 1),
                PointField('g', 13, PointField.UINT8, 1),
                PointField('b', 14, PointField.UINT8, 1),
            ]
            points = self.object_points
            cols = self.object_colors * 255
            # cols = cols.astype(np.uint8)
            # inlier_points = np.hstack((points, cols))
            # msg = pcl2.create_cloud(header, fields, inlier_points)
            msg = pcl2.create_cloud_xyz32(header, points)
            self.obj_cloud_pub.publish(msg)
        
        
        if visualize:
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(self.object_points)
            scene_pcd.colors = o3d.utility.Vector3dVector(self.object_colors)
            
            o3d.visualization.draw_geometries([scene_pcd])
        
    
    def get_grasp(self, visualize = False, publish = False):
        
        
        
        
        if self.object_points is None:
            return None
        
        # check if object is elongated on the table
        
        points = self.object_points
        
        # segment out the object and remove things that are not part of the object using depth information
        
        o3dpcd = o3d.geometry.PointCloud()
        o3dpcd.points = o3d.utility.Vector3dVector(points)
        o3dpcd.colors = o3d.utility.Vector3dVector(self.object_colors)
        
        labels = np.array(o3dpcd.cluster_dbscan(eps=0.01, min_points=50, print_progress=False))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")

        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        
        labels += 1
        
        cluster_sizes = np.bincount(labels)
        largest_cluster = np.argmax(cluster_sizes[1:]) + 1
        o3dpcd.points = o3d.utility.Vector3dVector(points[labels == largest_cluster])
        o3dpcd.colors = o3d.utility.Vector3dVector(self.object_colors[labels == largest_cluster])
        
        
        points = np.array(o3dpcd.points)
        
        ori_bbox = o3dpcd.get_oriented_bounding_box()
        axi_bbox = o3dpcd.get_axis_aligned_bounding_box()
        
        box_angles = np.array(ori_bbox.R)
        extents = np.array(ori_bbox.extent)
        
        long_axis = np.argmax(extents)
        unit_vecs_bbox = box_angles.T
        axis_vec = unit_vecs_bbox[long_axis]
        
        print(extents)
        
        
        
        grasp_center = np.array(ori_bbox.center)
        grasp_yaw = np.arctan2(axis_vec[1], axis_vec[0])
        # grasp_yaw = np.mod(grasp_yaw + 2 * np.pi, 2 * np.pi) - np.pi
        if grasp_yaw < 0:
            grasp_yaw += np.pi/2
        else:
            grasp_yaw -= np.pi/2    
        
        quat = R.from_matrix(box_angles).as_quat()
        
        if publish:
            rospy.loginfo("Publishing grasp pose")
            self.tf_broadcaster.sendTransform((grasp_center),
                tf.transformations.quaternion_from_euler(0, 0, grasp_yaw),
                rospy.Time.now(),
                "grasp_pose",
                "base_link"
            )
            # rospy.loginfo("Publishing grasp pose")
            # self.tf_broadcaster.sendTransform((grasp_center),
            #     quat,
            #     rospy.Time.now(),
            #     "grasp_pose",
            #     "base_link"
            # )
            
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"  # Set your desired frame_id
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                # rgb
                PointField('r', 12, PointField.UINT8, 1),
                PointField('g', 13, PointField.UINT8, 1),
                PointField('b', 14, PointField.UINT8, 1),
            ]
            points = np.array(o3dpcd.points)
            # cols = self.object_colors * 255
            # cols = cols.astype(np.uint8)
            # inlier_points = np.hstack((points, cols))
            # msg = pcl2.create_cloud(header, fields, inlier_points)
            msg = pcl2.create_cloud_xyz32(header, points)
            self.obj_cloud_pub.publish(msg)
        
        
        if visualize:
            line_set = o3d.geometry.LineSet()
            
            lineLength = max(h, w, l) / 2
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