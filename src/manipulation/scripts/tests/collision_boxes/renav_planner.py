
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import JointState
import open3d as o3d
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
import tf
import time
from std_msgs.msg import Header

import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PointStamped

def get_rotation_mat(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]], dtype = np.float64)

def visualize_boxes_rviz(boxes, centers):
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    for i, box in enumerate(boxes):
        center = np.array(box.center)
        extent = np.array(box.extent)
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 1
        marker.id = i

        # Set the scale of the marker
        marker.scale.x = extent[0] 
        marker.scale.y = extent[1]
        marker.scale.z = extent[2]

        # Set the color
        color = colors[i % len(colors)]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.5

        # Set the pose of the marker
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]
        ori = np.array(box.R)
        quat = R.from_matrix(ori).as_quat()
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker_pub.publish(marker)

def callback(msg):
    classes = [
        "joint_head_pan", 
        "joint_lift",
        "joint_arm_l0",
        "joint_arm_l1",
        "joint_arm_l2",
        "joint_arm_l3",
        "joint_wrist_yaw",
    ]
    
    
    
    final_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # x, y, theta, z, phi, ext
    for i, k in enumerate(msg.name):
        
        if 'arm' in k:
            final_state[5] += msg.position[i]
        if 'wrist' in k:
            final_state[4] += msg.position[i]
        if k == 'joint_lift':
            final_state[3] += msg.position[i]
        # if k == "joint_head_pan":
        #     final_state[2] -= msg.position[i]
    
    transform = get_transform(listener, 'link_arm_l4', 'link_arm_l0')
    final_state[5] = np.linalg.norm(transform[:3, 3])
    
    debug_state = {
        "x" : final_state[0],
        "y" : final_state[1],
        "theta" : final_state[2],
        "lift" : final_state[3],
        "arm" : final_state[4],
        "wrist" : final_state[5],
    }
    global curstate
    curstate = final_state
    # boxes, centers = get_collision_boxes(final_state)
    # visualize_boxes_rviz(boxes, centers)     
    # print(debug_state)

def get_transform(listener, base_frame, camera_frame):

    while not rospy.is_shutdown():
        try:
            t = listener.getLatestCommonTime(base_frame, camera_frame)
            (trans,rot) = listener.lookupTransform(base_frame, camera_frame, t)

            trans_mat = np.array(trans).reshape(3, 1)
            rot_mat = R.from_quat(rot).as_matrix()
            transform_mat = np.hstack((rot_mat, trans_mat))

            return transform_mat
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue



def parse_scene(depth, rgb, detection):
    global latest_detection
    global prevtime, rgbd_image, cam
    depth_img = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
    depth_image = np.array(depth_img, dtype=np.float32)

    color_img = bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")
    color_image = np.array(color_img, dtype=np.uint8)
    
    depth = depth_image.astype(np.uint16)
    depth[depth < 150] = 0
    depth[depth > 3000] = 0
    
    depth = o3d.geometry.Image(depth)
    color = o3d.geometry.Image(color_image.astype(np.uint8))
    
    extrinsics = get_transform(listener, 'camera_color_optical_frame', 'base_link')
    extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1000, convert_rgb_to_intensity=False
    )
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = intrinsics
    cam.width = color_image.shape[1]
    cam.height = color_image.shape[0]
    
    num_detections = (detection.nPredictions)
    msg = {
        "boxes" : np.array(detection.box_bounding_boxes).reshape(num_detections, 4),
        "box_classes" : np.array(detection.box_classes).reshape(num_detections),
        'confidences' : np.array(detection.confidences).reshape(num_detections),
    }

    loc = np.where(np.array(msg['box_classes']).astype(np.uint8) == 8)[0]


    if len(loc) > 0:
        loc = loc[0]
        box = np.squeeze(msg['boxes'][loc]).astype(int)
        confidence = np.squeeze(msg['confidences'][loc])
        x1, y1, x2, y2 =  box
        crop = depth_image[y1 : y2, x1 : x2]
                
        z_f = np.median(crop[crop != 0])/1000.0
        
        x_f = (x1 + x2)/2
        y_f = (y1 + y2)/2

        point = convertPointToFrame(x_f, y_f, z_f, "base_link")
        x, y, z = point.x, point.y, point.z
        latest_detection = [x, y, z]
        
def convertPointToFrame( x_f, y_f, z, to_frame = "base_link"):
    intrinsic = intrinsics
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
    point = listener.transformPoint(to_frame, camera_point).point
    return point

def capture_shot():
    global rgbd_image, cam, latest_detection, detections
    
    extrinsics = get_transform(listener, 'camera_color_optical_frame', 'base_link')
    extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)

    if latest_detection is not None:
        detections.append(latest_detection)
        
        detection = np.mean(detections, axis = 0)
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
        marker_pub.publish(marker)

    tsdf_volume.integrate(
            rgbd_image,
            cam,
            extrinsics)
    
    
def get_pcd():
    pcd = tsdf_volume.extract_point_cloud()
    points = np.array(pcd.points)
    points = points[points[:, 2] < 1.6]
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "base_link"  # Set your desired frame_id
    msg = pcl2.create_cloud_xyz32(header, points)
    obj_cloud_pub.publish(msg)
    print("publishing ")
    return pcd
    
if __name__ == '__main__':
    rospy.init_node('test')
    marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
    from helpers import move_to_pose
    from std_srvs.srv import Trigger, TriggerResponse
    import message_filters
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import CameraInfo
    from sensor_msgs.msg import PointCloud2
    import sys
    sys.path.append('/home/alfred/alfred-autonomy/src/manipulation/scripts/')
    sys.path.append('/home/hello-robot/alfred-autonomy/src/manipulation/scripts/planner')
    from yolo.msg import Detections
    from naive_planner import NaivePlanner
    from astar_planner import AStarPlanner
    from planner import Planner
    startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    startManipService.wait_for_service()
    startManipService()
    prevtime = time.time()
    import cv_bridge
    bridge = cv_bridge.CvBridge()
    
    scene_cloud = None
    rgbd_image = None
    cam = None
    latest_detection = None
    detections = []
    
    listener = tf.TransformListener()
    
    
    stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
    stow_robot_service.wait_for_service()
    trajectory_client = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    trajectory_client.wait_for_server()
    stow_robot_service()
    
    rospy.Subscriber('/alfred/joint_states', JointState, callback)
    move_to_pose(trajectory_client, {
        "head_pan;to" : -np.pi/2,
        "head_tilt;to" : -10 * np.pi/180,
    })
    rospy.sleep(2)
    
    depth_image_subscriber = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
    rgb_image_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image)
    detections_sub = message_filters.Subscriber('/object_bounding_boxes', Detections)
    
    obj_cloud_pub = rospy.Publisher('/object_cloud', PointCloud2, queue_size=10)

    msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=30)
    intrinsics = np.array(msg.K).reshape(3,3)

    synced_messages = message_filters.ApproximateTimeSynchronizer([
            depth_image_subscriber, rgb_image_subscriber,detections_sub
    ], 100, 0.5, allow_headerless=False)
    synced_messages.registerCallback(parse_scene)
    
    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length= 0.01,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )    
    
    rospy.loginfo("Waiting for stuff")
    while cam is None or rgbd_image is None:
        rospy.sleep(0.1)
        
    for i, angle in enumerate(np.linspace(-np.pi/2, np.pi/2, 13)):
        move_to_pose(trajectory_client, {
            'head_pan;to' : angle,
        })
        rospy.loginfo("Capturing shot %d" % i)
        rospy.sleep(0.8) #settling time.
        capture_shot()
        
    cloud = get_pcd()
    planner = Planner()
    planner.plan_base(cloud,np.mean(detections, axis = 0))
    
    
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

