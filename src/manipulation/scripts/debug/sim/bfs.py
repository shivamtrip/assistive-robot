from __future__ import annotations
import numpy as np

import open3d as o3d
import numpy as np
from collections import deque

class Node:
    def __init__(self, coord, parent=None):
        self.coord = coord
        self.parent = parent

class BFSPlanner:
    def __init__(self, occupancy_grid, start, goal):
        self.occupancy_grid = occupancy_grid
        self.start_node = Node(start)
        self.goal_node = Node(goal)
    
    def get_successors(self, node):
        x, y, z = node.coord
        successors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    new_x, new_y, new_z = x+dx, y+dy, z+dz
                    if (new_x >= 0 and new_x < self.occupancy_grid.shape[0] and
                        new_y >= 0 and new_y < self.occupancy_grid.shape[1] and
                        new_z >= 0 and new_z < self.occupancy_grid.shape[2] and
                        self.occupancy_grid[new_x, new_y, new_z] == 0):
                        successors.append(Node((new_x, new_y, new_z), parent=node))
        return successors
    
    def plan(self):
        frontier = deque([self.start_node])
        explored = set()
        
        while frontier:
            print(len(frontier))
            curr_node = frontier.popleft()
            
            if curr_node.coord == self.goal_node.coord:
                path = []
                while curr_node.parent is not None:
                    path.append(curr_node.coord)
                    curr_node = curr_node.parent
                path.append(curr_node.coord)
                return path[::-1]
            
            explored.add(curr_node.coord)
            
            for successor in self.get_successors(curr_node):
                if successor.coord not in explored:
                    frontier.append(successor)
                    explored.add(successor.coord)
        
        return None


  
def plot_plane(occupancyGrid, resolution, path):
    pointsFromOccupancy =  np.argwhere(occupancyGrid > 0)
    colors = (occupancyGrid[pointsFromOccupancy[:, 0], pointsFromOccupancy[:, 1], pointsFromOccupancy[:, 2]] * 0.8/occupancyGrid.max())
    print(colors.shape)
    # reshape 
    colors = np.stack((colors, colors, colors), axis = 1)
    pointsFromOccupancy = pointsFromOccupancy.astype(float)
    pointsFromOccupancy *= resolution
    plane_cloud = o3d.geometry.PointCloud()                 # creating point cloud of plane
    plane_cloud.points = o3d.utility.Vector3dVector(pointsFromOccupancy)
    # plane_cloud.colors = o3d.utility.Vector3dVector(np.ones((pointsFromOccupancy.shape[0], 3)))
    # color point cloud based on occupancy grid values  
    plane_cloud.colors = o3d.utility.Vector3dVector(colors)



    # app = gui.Application.instance
    # app.initialize()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(plane_cloud)

    # add path
    if path is not None:
        path = np.array(path).astype(float)
        path *= resolution
        path_cloud = o3d.geometry.PointCloud()
        path_cloud.points = o3d.utility.Vector3dVector(path)
        path_cloud.colors = o3d.utility.Vector3dVector(np.ones((path.shape[0], 3)) * np.array([1, 0, 0]))
        vis.add_geometry(path_cloud)


    opt = vis.get_render_option()
    opt.point_size = 20
    vis.run()

if __name__ == '__main__':
    ogrid = np.load('occupancyGrid_005.npy')
    ogrid = ogrid.transpose(1, 2, 0)
    occupancyGrid3d = ogrid.astype(bool).astype(int)
    start = (30, 0, 40)
    goal = (30, 30, 40)
    
    print(occupancyGrid3d[start[0], start[1], start[2]])
    print(occupancyGrid3d[goal[0], goal[1], goal[2]])
    planner = BFSPlanner(occupancyGrid3d, start, goal)
    # print(planner.plan())
    plot_plane(ogrid, 0.1, planner.plan())