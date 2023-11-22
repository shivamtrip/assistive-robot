import numpy as np
import open3d as o3d



class BasePlanner():
    
    def __init__(self, start, goal, point_cloud, l = [0.11587, 0.25, 0.25, 0.15]):
        self.lengths = l
        self.point_cloud = None
        
        self.start = start
        self.goal = goal
        self.pcd = point_cloud
        self.points_in_pcd = o3d.utility.Vector3dVector(np.array(self.pcd.points))
    
    def plan(self):
        # only method that is called from outside
        # takes in start, goal, and point cloud.
        # returns path that is waypoints from start to goal.
        raise NotImplementedError()
    
    
    def config_to_workspace(self, state):
    
        x, y, theta, z, phi, ext = state
        
        ee_x = x - self.lengths[0] * np.cos(theta) + (0.3 + ext - self.lengths[1]/2) * np.sin(theta) + self.lengths[2] * np.sin(theta + phi)
        ee_y = y - self.lengths[0] * np.sin(theta) - (0.3 + ext - self.lengths[1]/2) * np.cos(theta) - self.lengths[2] * np.cos(theta + phi)
        ee_z = z - self.lengths[3]    
        return ee_x, ee_y, ee_z
    
    def check_collision(self, state):
        boxes, _ = self.get_collision_boxes(state, get_centers = False)
        for box in boxes:
            if len(box.get_point_indices_within_bounding_box(self.points_in_pcd)) > 0:
                return True
        return False
    
    def check_valid_config(self, state):
        tx, ty, ttheta, tz, tphi, text = state
        
        if tz > 1.5 or tz < 0.1:
            return False    
        if text > 0.8 or text < 0:
            return False
        if ttheta > 2 * np.pi or ttheta < 0:
            return False
        if tphi > np.pi or tphi < -np.pi:
            return False
        if abs(tx) > 0.3 or abs(ty) > 0.3:
            return False
        
        if self.check_collision(state):
            return False
        
        return True
    
    def get_collision_boxes(self, state, get_centers = False):
        boxes = []
        x, y, theta, z, phi, ext = state
        l, l1, l3, _ = self.lengths
        centr = np.array([
            x - l * np.cos(theta), 
            y - l * np.sin(theta), 0.093621], dtype = np.float64).reshape(3, 1)
        base_box_o3d = o3d.geometry.OrientedBoundingBox(
            center = centr,
            R = BasePlanner.get_rotation_mat(theta).reshape(3, 3),
            extent = np.array([0.35, 0.35, 0.3], dtype = np.float64).reshape(3, 1)
        )
        
        boxes.append(base_box_o3d)
        
        arm_center = centr + np.array(
                [
                    (- l1 * np.sin(theta) + ( 0.3 + ext)* np.sin(theta))/2,
                    (l1 * np.cos(theta)  - ( 0.3 + ext) * np.cos(theta))/2,
                    z
                ]
            ).reshape(3, 1)
        arm_box_o3d = o3d.geometry.OrientedBoundingBox(
            center = arm_center,
            R = BasePlanner.get_rotation_mat(theta).reshape(3, 3),
            extent = np.array([0.1, ext + 0.3, 0.1], dtype = np.float64).reshape(3, 1)
        )
        
        boxes.append(arm_box_o3d)
        gripper_center = centr + np.array([
            (0.3 + ext - l1/2) * np.sin(theta) + l3/2 * np.sin(theta + phi),
            -(0.3 + ext - l1/2) * np.cos(theta) - l3/2 * np.cos(theta + phi),
            z - 0.15
        ]).reshape(3, 1)
        
        gripper_box_o3d = o3d.geometry.OrientedBoundingBox(
            center = gripper_center,
            R = BasePlanner.get_rotation_mat(theta + phi).reshape(3, 3),
            extent = np.array([0.15, 0.25, 0.15], dtype = np.float64).reshape(3, 1)
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
    
    
    
    @staticmethod
    def get_rotation_mat(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]], dtype = np.float64)