
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

def get_rotation_mat(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]], dtype = np.float64)

def get_collision_boxes(state, get_centers = False):
    x, y, theta, z, phi, ext = state
    boxes = []
    dist_baselink_to_center = 0.11587
    l1 = 0.25

    arm_min_extension = 0.3
    arm_dims = [0.1, ext + arm_min_extension, 0.1]
    
    gripper_length = 0.17
    gripper_dims = [0.17, 0.17, 0.18] # slightly recessed to allow grasp center to reach object.
    gripper_offset = [
        0.22, #distance from arm start to gripper start.
        0.09, # baseframe x offset to gripper hinge.
        -0.035, # basefram y offset to gripper hinge
        -gripper_dims[2] / 2, # gripper z offset
    ]

    base_z_offset = 0.093621
    base_box_dims = [0.33, 0.35, 0.19]

    base_center = np.array([
        x - dist_baselink_to_center * np.cos(theta), 
        y - dist_baselink_to_center * np.sin(theta), 
        base_z_offset
    ], dtype = np.float64).reshape(3, 1)
    
    base_box_o3d = o3d.geometry.OrientedBoundingBox(
        center = base_center,
        R = get_rotation_mat(theta).reshape(3, 3),
        extent = np.array(base_box_dims, dtype = np.float64).reshape(3, 1)
    )
    boxes.append(base_box_o3d)
    arm_center = np.array(
        [
            (- l1 * np.sin(theta) + (arm_min_extension + ext)* np.sin(theta))/2,
            (l1 * np.cos(theta)  - (arm_min_extension + ext) * np.cos(theta))/2,
            z + base_center[2][0]
        ]
    ).reshape(3, 1)
    arm_center += base_center
    
    arm_box_o3d = o3d.geometry.OrientedBoundingBox(
        center = arm_center,
        R = get_rotation_mat(theta).reshape(3, 3),
        extent = np.array(arm_dims, dtype = np.float64).reshape(3, 1)
    )
    
    boxes.append(arm_box_o3d)
    gripper_center = base_center + np.array([
        (gripper_offset[0] + ext - l1/2) * np.sin(theta) + (gripper_length/2 + 0) * np.sin(theta + phi) + gripper_offset[1],
        -(gripper_offset[0] + ext - l1/2) * np.cos(theta) - (gripper_length/2 + 0) * np.cos(theta + phi) + gripper_offset[2],
        z + base_center[2][0] + gripper_offset[3]
    ]).reshape(3, 1)
    
    gripper_box_o3d = o3d.geometry.OrientedBoundingBox(
        center = gripper_center,
        R = get_rotation_mat(theta + phi).reshape(3, 3),
        extent = np.array(gripper_dims, dtype = np.float64).reshape(3, 1)
    )
    boxes.append(gripper_box_o3d)
    centers = []
    if get_centers:
        center = o3d.geometry.TriangleMesh.create_sphere(radius=0.03).compute_vertex_normals()
        center.translate(base_center)
        center.paint_uniform_color([0, 0, 1])
        centers.append(center)
        center = o3d.geometry.TriangleMesh.create_sphere(radius=0.03).compute_vertex_normals()
        center.translate(np.array([x, y, 0.09]))
        center.paint_uniform_color([0, 0, 1])
        centers.append(center)
        center = o3d.geometry.TriangleMesh.create_sphere(radius=0.03).compute_vertex_normals()
        center.translate(gripper_center)
        center.paint_uniform_color([0, 0, 1])
        centers.append(center)
        center = o3d.geometry.TriangleMesh.create_sphere(radius=0.03).compute_vertex_normals()
        center.translate(arm_center)
        center.paint_uniform_color([0, 0, 1])
        centers.append(center)
    
    return boxes, centers

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


def parse_scene(depth, rgb ):
    global prevtime
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
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1000, convert_rgb_to_intensity=False
    )
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = intrinsics
    cam.width = color_image.shape[1]
    cam.height = color_image.shape[0]
    
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, cam, extrinsics
    )
    global scene_cloud
    
    cloud = cloud.voxel_down_sample(voxel_size=0.05)
    points = np.array(cloud.points)
    
    # scene_cloud = points.copy()
    scene_cloud = cloud
    if time.time() - prevtime > 2:
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_link"  # Set your desired frame_id
        msg = pcl2.create_cloud_xyz32(header, points)
        obj_cloud_pub.publish(msg)
        prevtime = time.time()
        print("publishing ")
    
def execute_plan(plan):
    waypoint = None
    # x, y, theta, z, phi, ext
    waits = [0.5, 0.1, 0.5, 0.5, 0.1, 0.5]
    for i, waypoint in enumerate(plan):
        boxes, centers = planner.get_collision_boxes(waypoint)
        visualize_boxes_rviz(boxes, centers)     
        rospy.loginfo("Sending waypoint")
        idx_to_move = 0
        for j in range(len(waypoint)):
            if waypoint[j] != plan[i-1][j]:
                idx_to_move = j
                break
        move_to_pose(trajectory_client, {
            "base_translate;by" : waypoint[0],
            "lift|;to" : waypoint[3],
            "arm|;to" : waypoint[5],
            "wrist_yaw|;to": waypoint[4],
        })
        rospy.sleep(waits[idx_to_move])
        
    if waypoint is None:
        return
    move_to_pose(trajectory_client, {
        "lift|;to" : waypoint[3],
        "arm|;to" : waypoint[5],
        "wrist_yaw|;to": waypoint[4],
    })
        


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
    from naive_planner import NaivePlanner
    from astar_planner import AStarPlanner
    
    startManipService = rospy.ServiceProxy('/switch_to_manipulation_mode', Trigger)
    startManipService.wait_for_service()
    startManipService()
    prevtime = time.time()
    import cv_bridge
    bridge = cv_bridge.CvBridge()
    
    scene_cloud = None
    
    listener = tf.TransformListener()
    
    
    stow_robot_service = rospy.ServiceProxy('/stow_robot', Trigger)
    stow_robot_service.wait_for_service()
    trajectory_client = actionlib.SimpleActionClient('alfred_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    trajectory_client.wait_for_server()
    stow_robot_service()
    
    rospy.Subscriber('/alfred/joint_states', JointState, callback)
    move_to_pose(trajectory_client, {
        "head_pan;to" : -np.pi/2,
        "head_tilt;to" : -30 * np.pi/180,
    })
    rospy.sleep(2)
    
    depth_image_subscriber = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
    rgb_image_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image)
    
    obj_cloud_pub = rospy.Publisher('/object_cloud', PointCloud2, queue_size=10)

    msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=30)
    intrinsics = np.array(msg.K).reshape(3,3)

    synced_messages = message_filters.ApproximateTimeSynchronizer([
            depth_image_subscriber, rgb_image_subscriber,
    ], 100, 0.5, allow_headerless=False)
    synced_messages.registerCallback(parse_scene)
    
    
    # x, y, theta, z, phi, ext
    curstate = None
    while curstate is None or scene_cloud is None:
        rospy.loginfo("Waiting for populating vars")
        rospy.sleep(0.1)
        
    rospy.sleep(2)
    
    curstate[2] = 0
    curstate[4] = 1 #stowed
    print("START", curstate)
    # exit()
    goalstate = [0, 0.0, 0, 0.8, 0, 0.6]
    planner = AStarPlanner(curstate, goalstate, scene_cloud)
    
    plan, success = planner.plan()
    print("Plan: ", plan)
    print("Success: ", success)

    # x, y, theta, z, phi, ext
    
    execute_plan(plan)
    rospy.sleep(2)
    plan = plan[::-1]
    execute_plan(plan)
    
    # stow_robot_service()
        
        
        
    
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

