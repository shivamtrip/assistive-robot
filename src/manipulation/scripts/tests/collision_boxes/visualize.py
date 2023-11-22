
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
    boxes = []
    lengths = [0.11587, 0.25, 0.17, 0.15]
    x, y, theta, z, phi, ext = state
    l, l1, l3, _ = lengths
    centr = np.array([
        x - l * np.cos(theta), 
        y - l * np.sin(theta), 0.093621], dtype = np.float64).reshape(3, 1)
    base_box_o3d = o3d.geometry.OrientedBoundingBox(
        center = centr,
        R = get_rotation_mat(theta).reshape(3, 3),
        extent = np.array([0.35, 0.35, 0.2], dtype = np.float64).reshape(3, 1)
    )
    boxes.append(base_box_o3d)
    arm_center = centr + np.array(
            [
                (- l1 * np.sin(theta) + ( 0.3 + ext)* np.sin(theta))/2,
                (l1 * np.cos(theta)  - ( 0.3 + ext) * np.cos(theta))/2,
                z + centr[2]
            ]
        ).reshape(3, 1)
    arm_box_o3d = o3d.geometry.OrientedBoundingBox(
        center = arm_center,
        R = get_rotation_mat(theta).reshape(3, 3),
        extent = np.array([0.1, ext + 0.3, 0.1], dtype = np.float64).reshape(3, 1)
    )
    
    boxes.append(arm_box_o3d)
    gripper_center = centr + np.array([
        (0.22 + ext - l1/2) * np.sin(theta) + (l3/2 + 0) * np.sin(theta + phi) + 0.09,
        -(0.22 + ext - l1/2) * np.cos(theta) - (l3/2 + 0) * np.cos(theta + phi) - 0.035,
        z + centr[2] - 0.075
    ]).reshape(3, 1)
    
    gripper_box_o3d = o3d.geometry.OrientedBoundingBox(
        center = gripper_center,
        R = get_rotation_mat(theta + phi).reshape(3, 3),
        extent = np.array([0.12, l3, 0.18], dtype = np.float64).reshape(3, 1)
    )
    boxes.append(gripper_box_o3d)
    centers = []
    if get_centers:
        center = o3d.geometry.TriangleMesh.create_sphere(radius=0.03).compute_vertex_normals()
        center.translate(centr)
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

