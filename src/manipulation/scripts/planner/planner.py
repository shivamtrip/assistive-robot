import numpy as np
import cv2
import open3d as o3d
from planner.astar_planner import AStarPlanner
from planner.naive_planner import NaivePlanner

class Planner:
    
    
    def __init__(self):

        self.resolution = 0.05 #resolution for planning base and voxelizing/rasterizing point cloud
        self.radius_reachable = int(0.85 / self.resolution)
        self.max_dist_traverse = int(1.3 / self.resolution)
        self.characteristic_dist = 0.15 / self.resolution
        
        self.lengths = [
            0.11587, 0.25, 0.25
        ]
    
    def plan_base(self, point_cloud, object_location, viz = False):
        points = np.array(point_cloud.points)
        x_proj = points[:,0]
        y_proj = points[:,1]

        x_min = np.min(x_proj)
        x_max = np.max(x_proj)
        y_min = np.min(y_proj)
        y_max = np.max(y_proj)


        source = np.array([0.0, 0.0])
        target = object_location

        x_grid = np.arange(x_min - 1.0, x_max + 1.0, self.resolution)
        y_grid = np.arange(y_min - 1.0, y_max + 1.0, self.resolution)

        img = np.zeros((len(x_grid), len(y_grid)))
        h, w = img.shape[:2]
        sorted_points = points[np.argsort(points[:,2])]

        def convert_point_to_img(x, y, offset = 0):
            x_ind = int((x - x_min)/self.resolution) + offset
            y_ind = int((y - y_min)/self.resolution) + offset
            return x_ind, y_ind
            
        def convert_point_to_world(x, y, offset = 0):
            x = (x - offset) * self.resolution + x_min
            y = (y - offset) * self.resolution + y_min
            return x, y

        for point in sorted_points:
            x, y, z = point
            x_ind, y_ind = convert_point_to_img(x, y)
            img[x_ind, y_ind] = z

        img -= np.min(img)
        img /= np.max(img)
        binary = (img <= 0.1) * 255
        img *= 255

        safety_dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        reachability_map = np.zeros_like(img[:, :, 0], dtype = np.uint8)
        source_reachability_map = np.zeros_like(reachability_map)

        cv2.circle(reachability_map, convert_point_to_img(target[0], target[1])[::-1], self.radius_reachable, 255, -1)
        cv2.circle(source_reachability_map, convert_point_to_img(source[0], source[1])[::-1], self.max_dist_traverse, 255, -1)

        target_reachable_dist = cv2.distanceTransform(reachability_map, cv2.DIST_L2, 5)
        source_reachabile_dist = cv2.distanceTransform(source_reachability_map, cv2.DIST_L2, 5)

        safety_dist[safety_dist < self.characteristic_dist] = 0

        reachability_map = target_reachable_dist * source_reachabile_dist * safety_dist
        
        is_plan_reqd = True
        if reachability_map[convert_point_to_img(0, 0)] != 0:
            is_plan_reqd = False
        
        reachability_map -= np.min(reachability_map)
        reachability_map /= np.max(reachability_map)
        reachability_map[reachability_map < 0.8] = 0
        
        
        
        # pick the point with the highest distance
        base_loc = np.unravel_index(np.argmax(reachability_map), reachability_map.shape)
        img[:, :, 2][reachability_map != 0] = reachability_map[reachability_map!=0] * 255
        img = cv2.circle(img, (base_loc[0], base_loc[1])[::-1], 1, (0, 255, 255), -1)


        img = cv2.circle(img, convert_point_to_img(source[0], source[1])[::-1], 2, (0, 255, 0), -1)
        img = cv2.circle(img, convert_point_to_img(target[0], target[1])[::-1], self.radius_reachable, (255, 0, 0), 1)
        img = cv2.circle(img, convert_point_to_img(target[0], target[1])[::-1], 2, (255, 0, 0), -1)
        img = cv2.circle(img, convert_point_to_img(source[0], source[1])[::-1], self.max_dist_traverse, (0, 255, 0), 1)

        cv2.imwrite("/home/hello-robot/alfred-autonomy/src/manipulation/scripts/tests/collision_boxes/reachability_map.png", img)        
        # we want base_dir to be perpendicular to the line joining base_loc and target
        
        base_loc = convert_point_to_world(base_loc[0], base_loc[1])
        base_theta = np.arctan2(-(target[0] - base_loc[0]), (target[1] - base_loc[1]))
        navigation_target = np.array(
            [
                base_loc[0],
                base_loc[1],
                0,
                base_theta
            ]
        )
        return navigation_target, is_plan_reqd
    
    
    
    def get_state_from_ee_target(self, ee_target):
        tx, ty, tz = ee_target
        ttheta = np.arctan2(ty, tx)
        tphi = np.pi/2 - np.arctan2(tz, np.sqrt(tx**2 + ty**2))
        text = np.sqrt(tx**2 + ty**2 + tz**2) - 0.3
        return np.array([tx, ty, ttheta, tz, tphi, text])
    

    def plan_manipulator(self, point_cloud, curstate, ee_target):
        
        state_target = self.get_state_from_ee_target(ee_target)
        
        # path, success = NaivePlanner(curstate, state_target, point_cloud).plan()

        # if success:
        #     return path, True
        
        path, success = AStarPlanner(curstate, state_target, point_cloud).plan()
        
        if success:
            return path, True
        else:
            return [], False
            
    
    def plan_grasp_target(self, point_cloud, object_bbox):
        o3d_box = o3d.geometry.OrientationBoundingBox.create_from_points(o3d.utility.Vector3dVector(object_bbox))
        cropped_cloud = point_cloud.crop(o3d_box)
        
        object_points = np.asarray(cropped_cloud.points)
        medianz = np.median(object_points[:,2])
        x, y = np.mean(object_points[object_points[:,2] == medianz, :2], axis=0)

        median_grasp = np.array([x, y, medianz])
        
        return median_grasp
                
    @staticmethod
    def get_rotation_mat(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]], dtype = np.float64)
        
        
    
