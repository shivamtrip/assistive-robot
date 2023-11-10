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
import message_filters
import open3d as o3d


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
        
        self.depth_image_subscriber = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.rgb_image_subscriber = rospy.Subscriber('/camera/color/image_raw', Image)
        synced = message_filters.ApproximateTimeSynchronizer([self.depth_image_subscriber, self.rgb_image_subscriber], 10, 0.1, allow_headerless=True)
        synced.registerCallback(self.image_callback)
        
        self.boundingBoxSub = rospy.Subscriber("/object_bounding_boxes", Detections, self.parse_detections)
        
        self.class_list = rospy.get_param('class_list')
        self.graph : Graph = Graph(len(self.class_list))
        
        
        
        while not self.obtainedInitialImages:
            rospy.sleep(0.1)
        
        
        rospy.Service('/clear_scene', Trigger, self.clear_graph)
        # TODO add custom service to start tracking object
        rospy.Service('/start_tracking_object', Trigger, self.start_tracking_object)
        rospy.Service('/stop_tracking_object', Trigger, self.stop_tracking_object)
        
        rospy.Service('/start_accumulate_tsdf', Trigger, self.start_accumulate_tsdf)
        rospy.Service('/stop_accumulate_tsdf', Trigger, self.stop_accumulate_tsdf)
        
        rospy.loginfo(f"[{self.node_name}]: Initial images obtained.")
        rospy.loginfo(f"[{self.node_name}]: Node initialized")
        
        self.trackedNode = None
        self.tsdf_volume = None
        self.point_cloud = None
    
    def start_accumulate_tsdf(self, req):
        self.accumulating_tsdf = True
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        return True, "Accumulated TSDF"
    
    def stop_accumulate_tsdf(self, req):
        self.accumulating_tsdf = False
        pcd = self.tsdf_volume.extract_point_cloud()
        self.point_cloud = pcd
        o3d.io.write_point_cloud("/home/hello-robot/alfred-autonomy/src/perception/uatu/outputs/tsdf.ply", pcd)
        return True, "Accumulated TSDF"
    
    def start_tracking_object(self, req):
        self.trackedNode = Node(req.objectId)
        
    def clear_graph(self, req):
        self.graph.clear()
        rospy.loginfo(f"[{self.node_name}]: Graph cleared.")

    def tsdf_snapshot(self, req):
        if not self.accumulating_tsdf:
            return False
        
        depth = self.depth_image.astype(np.uint16)
        depth[depth < 150] = 0
        depth[depth > 3000] = 0
        
        depth = o3d.geometry.Image(depth)
        color = o3d.geometry.Image(self.color_image.astype(np.uint8))
        
        extrinsics = self.get_transform('/camera_color_optical_frame', '/base_link')
        extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)
        intrinsics = self.intrinsics
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000, convert_rgb_to_intensity=False
        )
        cam = o3d.camera.PinholeCameraIntrinsic()
        cam.intrinsic_matrix = intrinsics
        cam.width = self.color.shape[1]
        cam.height = self.color.shape[0]
        self.tsdf_volume.integrate(
            rgbd,
            cam,
            extrinsics)
        return True

    def image_snapshot(self):
        extrinsics_to_base = self.get_transform('/camera_color_optical_frame', '/base_link')
        extrinsics_to_base = np.concatenate((extrinsics_to_base, np.array([[0, 0, 0, 1]])), axis=0)
        
        extrinsics_to_map = self.get_transform('/camera_color_optical_frame', '/map')
        extrinsics_to_map = np.concatenate((extrinsics_to_map, np.array([[0, 0, 0, 1]])), axis=0)
        
        detections = self.latest_detections
        classes = detections['box_classes']
        indices = classes == self.trackedNode.cls_type
        boxes = detections['boxes'][indices]
        
        if len(boxes) != 0:
            self.trackedNode.add_sample(
                self.color_image, self.depth_image,
                extrinsics_to_base, extrinsics_to_map,
                boxes
            )

    def image_callback(self, depth_img, color_img):
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        self.depth_image = np.array(depth_img, dtype=np.uint16)
        self.color_image = self.bridge.imgmsg_to_cv2(color_img, desired_encoding="passthrough")
            
    def get_three_d_bbox_from_tracker(self):
        if self.trackedNode == None or self.point_cloud == None:
            return False
        
        cloud = self.trackedNode.reconstruct_object()
        
        bbox = cloud.get_oriented_bounding_box()
        box_points = np.array(bbox.get_box_points())
        
        return box_points
        
        

    def convert_point_to_base_frame(self, x_f, y_f, z):
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

    def parse_detections(self, msg : Detections):
        self.obtainedInitialImages = True
        num_detections = (msg.nPredictions)
        msg = {
            "boxes" : np.array(msg.box_bounding_boxes).reshape(num_detections, 4),
            "box_classes" : np.array(msg.box_classes).reshape(num_detections),
            'confidences' : np.array(msg.confidences).reshape(num_detections),
        }
        self.latest_detections = msg
        n = len(msg['box_classes'])
        
        upscale_fac = self.upscale_fac
        for i in range(n):
            box = np.squeeze(msg['boxes'][n])
            confidence = np.squeeze(msg['confidences'][n])
            class_type = np.squeeze(msg['box_classes'][n])
            x1, y1, x2, y2 =  box
            mask = np.zeros_like(self.depth_image)
            mask[int(y1):int(y2), int(x1):int(x2)] = 1
            depth = self.depth_image * mask
            color = self.color_image * mask
            
            self.graph.add_observation(
                color, depth,
                class_type, confidence, 
                msg.header.stamp)

if __name__ == "__main__":
    node = SceneParser()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass