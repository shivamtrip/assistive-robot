from __future__ import annotations
import numpy as np


import heapq
import numpy as np

class Node:
    def __init__(self, coord, g=0, h=0):
        self.coord = coord
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = None
    
    def __lt__(self, other):
        return self.f < other.f
    
class AStarPlanner:
    def __init__(self, occupancy_grid, start, goal):
        self.occupancy_grid = occupancy_grid
        self.start_node = Node(start)
        self.goal_node = Node(goal)
    
    def heuristic(self, node):
        dx = self.goal_node.coord[0] - node.coord[0]
        dy = self.goal_node.coord[1] - node.coord[1]
        dz = self.goal_node.coord[2] - node.coord[2]
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
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
                        successors.append(Node((new_x, new_y, new_z)))
        return successors
    
    def plan(self):
        open_list = [self.start_node]
        closed_list = []
        
        while len(open_list) > 0:
            print(len(open_list))
            curr_node = heapq.heappop(open_list)
            
            if curr_node.coord == self.goal_node.coord:
                path = []
                while curr_node is not None:
                    path.append(curr_node.coord)
                    curr_node = curr_node.parent
                return path[::-1]
            
            closed_list.append(curr_node)
            
            for successor in self.get_successors(curr_node):
                if successor in closed_list:
                    continue
                
                new_g = curr_node.g + self.heuristic(successor)
                if successor not in open_list:
                    successor.g = new_g
                    successor.h = self.heuristic(successor)
                    successor.f = successor.g + successor.h
                    successor.parent = curr_node
                    heapq.heappush(open_list, successor)
                elif new_g < successor.g:
                    successor.g = new_g
                    successor.f = successor.g + successor.h
                    successor.parent = curr_node
                    heapq.heapify(open_list)
        
        return None



if __name__ == '__main__':
    occupancyGrid3d = np.load('occupancyGrid.npy').astype(bool).astype(int)
    print(occupancyGrid3d.shape)
    print(occupancyGrid3d[0, 0, 0])
    print(occupancyGrid3d[10, 10, 10])
    planner = AStarPlanner(occupancyGrid3d, (0, 0, 0), (10, 10, 10))
    planner.plan()