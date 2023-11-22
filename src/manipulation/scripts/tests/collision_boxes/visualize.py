
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import JointState
import open3d as o3d
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R


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

    arm_center = base_center + np.array(
        [
            (- l1 * np.sin(theta) + (arm_min_extension + ext)* np.sin(theta))/2,
            (l1 * np.cos(theta)  - (arm_min_extension + ext) * np.cos(theta))/2,
            z + base_center[2]
        ]
    ).reshape(3, 1)
    
    arm_box_o3d = o3d.geometry.OrientedBoundingBox(
        center = arm_center,
        R = get_rotation_mat(theta).reshape(3, 3),
        extent = np.array(arm_dims, dtype = np.float64).reshape(3, 1)
    )
    
    boxes.append(arm_box_o3d)
    gripper_center = base_center + np.array([
        (gripper_offset[0] + ext - l1/2) * np.sin(theta) + (gripper_length/2 + 0) * np.sin(theta + phi) + gripper_offset[1],
        -(gripper_offset[0] + ext - l1/2) * np.cos(theta) - (gripper_length/2 + 0) * np.cos(theta + phi) + gripper_offset[2],
        z + base_center[2] + gripper_offset[3]
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
            
    debug_state = {
        "x" : final_state[0],
        "y" : final_state[1],
        "theta" : final_state[2],
        "lift" : final_state[3],
        "arm" : final_state[4],
        "wrist" : final_state[5],
    }
    boxes, centers = get_collision_boxes(final_state)
    visualize_boxes_rviz(boxes, centers)
    # print(msg.name)
    # print(msg.position)
    print(debug_state)

if __name__ == '__main__':
    rospy.init_node('test')
    rospy.Subscriber('/alfred/joint_states', JointState, callback)
    marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

