import numpy as np
import cv2

class Planner:
    
    
    def __init__(self):
        self.get_segmentation_from_bounds = None
        self.resolution = 0.05 #resolution for planning base and voxelizing/rasterizing point cloud
        self.radius_reachable = int(0.85 / self.resolution)
        self.max_dist_traverse = int(1.3 / self.resolution)
        self.characteristic_dist = 0.15 / self.resolution
    
    
    def plan_base(self, point_cloud, object_location, viz = False):
        points = point_cloud
        x_proj = points[:,0]
        y_proj = points[:,1]

        x_min = np.min(x_proj)
        x_max = np.max(x_proj)
        y_min = np.min(y_proj)
        y_max = np.max(y_proj)


        source = np.array([0.0, 0.0])
        target = object_location

        x_grid = np.arange(x_min, x_max, self.resolution)
        y_grid = np.arange(y_min, y_max, self.resolution)

        img = np.zeros((len(x_grid), len(y_grid)))
        sorted_points = points[np.argsort(points[:,2])]

        def convert_point(x, y):
            x_ind = int((x - x_min)/self.resolution)
            y_ind = int((y - y_min)/self.resolution)
            return x_ind, y_ind
            

        for point in sorted_points:
            x, y, z = point
            x_ind, y_ind = convert_point(x, y)
            img[x_ind, y_ind] = z

        img -= np.min(img)
        img /= np.max(img)
        binary = (img <= 0.1) * 255
        img *= 255

        safety_dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        reachability_map = np.zeros_like(img[:, :, 0], dtype = np.uint8)
        source_reachability_map = np.zeros_like(reachability_map)

        cv2.circle(reachability_map, convert_point(target[0], target[1])[::-1], self.radius_reachable, 255, -1)
        cv2.circle(source_reachability_map, convert_point(source[0], source[1])[::-1], self.max_dist_traverse, 255, -1)

        target_reachable_dist = cv2.distanceTransform(reachability_map, cv2.DIST_L2, 5)
        source_reachabile_dist = cv2.distanceTransform(source_reachability_map, cv2.DIST_L2, 5)

        safety_dist[safety_dist < self.characteristic_dist] = 0

        reachability_map = target_reachable_dist * source_reachabile_dist * safety_dist
        reachability_map -= np.min(reachability_map)
        reachability_map /= np.max(reachability_map)
        reachability_map[reachability_map < 0.8] = 0

        # pick the point with the highest distance
        base_loc = np.unravel_index(np.argmax(reachability_map), reachability_map.shape)
        base_theta = np.arctan2(-(target[0] - base_loc[0]), (target[1] - base_loc[1]))

        # we want base_dir to be perpendicular to the line joining base_loc and target
        
        base_loc = convert_point(base_loc[0], base_loc[1])
        navigation_target = np.array(
            [
                base_loc[0],
                base_loc[1],
                base_theta
            ]
        )
        
        return navigation_target

    
    def plan_manipulator(self, target):
        pass
    
    def plan_grasp_target(self, point_cloud, object_location):
        segmentation = self.get_segmentation_from_bounds(point_cloud, object_location)
        
        # choose centroid
        
        points_of_interest = point_cloud[segmentation]
        
        # choose grasp target 
        
        
    
