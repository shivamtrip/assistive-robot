#!/usr/local/lib/robot_env/bin/python3

import argparse
from collections import OrderedDict 
import os
import rospy
import cv2
import json
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import OccupancyGrid
import numpy as np
import matplotlib.pyplot as plt
import cv2
class ProcessCostmap():

    def __init__(self, update_callback):

        self.goal = None
        self.update_callback = update_callback
        self.costmap = None
        self.costmap_info = None
        
        self.costmap_sub = rospy.Subscriber(rospy.get_param('/navigation/global_costmap_topic'), OccupancyGrid, self.costmap_callback)
        # self.costmap = 
        while self.costmap is None and not rospy.is_shutdown():
            rospy.loginfo("Waiting for global costmap...")
            rospy.sleep(1)
        rospy.loginfo("Received global costmap...")
        
        # Obstacle threshold
        self.des_obs_thresh = rospy.get_param('/navigation/des_obs_thresh')
        self.big_obs_thresh = rospy.get_param('/navigation/big_obs_thresh')
        

        rospy.loginfo("Costmap processor initialized!!")
        
        
        
        
    def set_goal(self, goal):
        self.goal = goal

    def costmap_callback(self, msg):
        """ Callback function to receive costmap """

        self.costmap = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.costmap_info = msg.info
        if self.goal is not None:
            safe = self.isSafe(self.goal.x, self.goal.y, 0.0)
            self.goal = None
            if not safe:
                self.update_callback()
        # self.viz_cv(0, 0)
        # visualizations
        
    def viz_cv(self, x, y):
        costmap = self.costmap
        costmap = costmap.astype(np.float32)
        costmap -= np.min(costmap)
        costmap /= np.max(costmap)
        costmap *= 255
        costmap = costmap.astype(np.uint8)
        costmap = cv2.cvtColor(costmap, cv2.COLOR_GRAY2BGR)

        print("COLORING", x, y)
        cv2.circle(costmap, (int(x), int(y)), 5, (0, 0, 255), -1)
        mask = costmap > 10
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        costmap = costmap[rmin:rmax,cmin:cmax]
        costmap = cv2.resize(costmap, (0,0), fx=3, fy=3)
        cv2.imwrite('/home/hello-robot/alfred-autonomy/src/navigation/alfred_navigation/scripts/tests/temp.png', costmap)
        # cv2.imshow("costmap", costmap)
        # cv2.waitKey(1)

    def print_map_params(self,):
        """ Print the map parameters"""

        assert self.costmap is not None  ," Empty costmap!! Please load costmap first!!"

        info = self.costmap_info    
        print("Map parameters: ")
        print("Origin: ",info.origin)
        print("Resolution: ",info.resolution)
        print("Width: ",info.width)
        print("Height: ",info.height)
        print("Data: ",self.costmap.shape)

    def get_cell_coordinates(self,row, col):
        """ Given cell coordinates in costmap, return the corresponding world coordinates"""
        
        assert self.costmap is not None ," Empty costmap!!"

        info = self.costmap_info    
        x = info.origin.position.x + col * info.resolution
        y = info.origin.position.y + row * info.resolution
        z = info.origin.position.z  # z=0 for 2D costmaps

        return x, y, z

    def isValidRowCol(self,row,col):
        
        assert self.costmap is not None ," Empty costmap!!Please load costmap first!!"

        info = self.costmap_info    
        if row >= 0 and row < info.height and col >= 0 and col < info.width:
            return True
        else:
            return False

    def get_cell_indices(self,x, y,z=0):
        """ Given world coordinates, return the corresponding cell coordinates in costmap"""

        assert self.costmap is not None ," Empty costmap!! Please load costmap first!!"
        assert np.isclose(z,0) ," Costmap is 2D but z is not 0!!"

        info = self.costmap_info    
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)

        assert self.isValidRowCol(row,col), "Computed cell indices out of bounds!! Given x,y,z: {},{},{} are bad points!".format(x,y,z)
        return row, col


    def isSafe(self,x,y,z):

        """ Given world coordinates, return whether the cell is safe or not. It helps dowstream in determining whether path exists or not"""

        assert self.costmap is not None ," Empty costmap!! Please load costmap first!!"

        max_allowed_cost = 50
        safe_cost = 30

        row, col = self.get_cell_indices(x, y)
        assert self.isValidRowCol(row,col), "Computed cell indices out of bounds!! Given x,y,z: {},{},{} are bad points!".format(x,y,z)

        if row is not None and col is not None:
            if self.costmap[row,col] <= safe_cost:
                rospy.loginfo("Cell is safe!! Row: {}, Col: {}".format(row,col))
                return True
            elif self.costmap[row,col] > safe_cost and self.costmap[row,col] < max_allowed_cost:
                rospy.loginfo("Cell Row: {}, Col: {} has significant cost {}".format(row,col,self.costmap[row,col]))
                return True
            else:
                rospy.logdebug("Cell is not safe!! Row: {}, Col: {}. Cost {}".format(row,col,self.costmap[row,col]))
                return False
        else:
            rospy.logwarn("No cell indices computed!!")
            return False

    def findNearestSafeCell(self,x,y,z):
        """ Given world coordinates, return the nearest safe cell coordinates in costmap"""

        assert self.costmap is not None ," Empty costmap!! Please load costmap first!!"

        row, col = self.get_cell_indices(x, y)
        assert self.isValidRowCol(row,col), "Computed cell indices out of bounds!! Given x,y,z: {},{},{} are bad points!".format(x,y,z)

        if row is not None and col is not None:
            if self.isSafe(x,y,z):
                rospy.logdebug("Computed Cell is safe!! Row: {}, Col: {}".format(row,col))
                return row, col
            else:
                rospy.logdebug("Computed Cell is not safe!! Row: {}, Col: {}".format(row,col))
                return self.RunBFSForSearch(row,col)
        else:
            rospy.logwarn("No cell indices computed!!")
            return None

    def findNearestSafePoint(self,x,y,z):
        """ Given world coordinates, return the nearest safe cell coordinates in costmap"""

        assert self.costmap is not None ," Empty costmap!! Please load costmap first!!"

        row, col = self.get_cell_indices(x, y)
        print("Retrieved row, col: ",(row,col))
        assert self.isValidRowCol(row,col), "Computed cell indices out of bounds!! Given x,y,z: {},{},{} are bad points!".format(x,y,z)

        if row is not None and col is not None:
            if self.isSafe(x,y,z):
                rospy.loginfo("Computed Cell is safe!! Row: {}, Col: {}".format(row,col))
                return x, y, z
            else:
                rospy.loginfo("Computed Cell is not safe!! Row: {}, Col: {}".format(row,col))
                row,col = self.RunBFSForSearch(row, col)
                # self.viz_cv(col, row)
                coords = self.get_cell_coordinates(row, col)
                return coords
        else:
            rospy.logwarn("No cell indices computed!!")
            return None
    
    def RunBFSForSearch(self,row,col):
        """ Given cell coordinates, return the nearest safe cell coordinates in costmap"""

        assert self.costmap is not None ," Empty costmap!! Please load costmap first!!"

        rospy.logdebug("Running nearest safe cell finder!!")

        # RUN BFS on the costmap near the given cell to find the nearest safe cell
        info = self.costmap_info    
        queue = [(row,col)]
        visited = set()
        visited.add((row,col))
        while queue:
            row,col = queue.pop(0)
            if self.isSafe(*self.get_cell_coordinates(row,col)):
                rospy.loginfo("Found nearest safe cell!! Row: {}, Col: {}".format(row,col))
                return row,col
            else:
                for i in range(-1,2):
                    for j in range(-1,2):
                        if (row+i,col+j) not in visited and self.isValidRowCol(row+i,col+j):
                            queue.append((row+i,col+j))
                            visited.add((row+i,col+j))
        rospy.logwarn("No safe cell found!!")
        
    def sample_obstacle_cells(self,ref_cost=50,num_samples=1,num_tries=1000):

        """ Sample grid cells in costmap that is an obstacle"""

        rospy.loginfo("Sampling obstacle cells!!")
            
        assert self.costmap is not None ," Empty costmap!! Please load costmap first!!"

        iters = 0
        samples=[]

        mask = self.costmap < ref_cost
        # find largest bounding box where mask is true
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        if self.LOADED_COSTMAP:
            info = self.costmap_info    
            for i in range(num_samples):
                rospy.loginfo("Sampling {}th obstacle point!!".format(i))
                while iters<num_tries:
                    row = np.random.randint(rmin,rmax)
                    col = np.random.randint(cmin,cmax)
                    print("Cost sampled: ",self.costmap[row,col])
                    if self.costmap[row,col] > ref_cost:
                        samples.append((row,col))
                        break

        return np.array(samples)
    
    def normalize(self,row,col):

        row = row - self.rmin
        col = col - self.cmin

        return row,col

        
    def visualize_and_debug(self,):
        """ Useful for visualization and debugging all helper functions"""

        rospy.loginfo("Visualizing and debugging!!")
        plt.ion()
        plt.clf()
        plt.cla()
        # Print the map parameters
        self.print_map_params()

        origin = (0,0,0)
        num_samples = 1
        ref_cost = self.big_obs_thresh
        obstacle_cells = self.sample_obstacle_cells(num_samples=num_samples,ref_cost=ref_cost)
        obstacle_points = [self.get_cell_coordinates(obstacle_cells[i,0],obstacle_cells[i,1]) for i in range(num_samples)]
        
        # Center of 'sofa' in map
        obstacle_points = [(-3.35,-7.49,0)]

        # Find nearest safe points
        rospy.loginfo("Finding nearest safe points!!")
        safe_points = []
        for i in range(num_samples):
            safe_points.append(self.findNearestSafePoint(*obstacle_points[i]))

        # Make a color palette for matplotlib colors 
        colors = ['red','green','blue','yellow','orange','purple','pink','brown','gray','olive','cyan','magenta','black','white']

        # Trim the costmap to region that matters
        mask = self.costmap < ref_cost
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        trimmed_costmap = self.costmap[rmin:rmax,cmin:cmax]

        # Store in class variables
        self.rmin = rmin
        self.rmax = rmax
        self.cmin = cmin
        self.cmax = cmax

        fig,ax = plt.subplots()
        ax.imshow(trimmed_costmap/100,cmap='gray')
        for i in range(num_samples):
            r,c = self.normalize(*self.get_cell_indices(*obstacle_points[i]))
            ro,co = self.normalize(*self.get_cell_indices(*origin))
            rs,cs = self.normalize(*self.get_cell_indices(*safe_points[i]))
        
            ax.scatter(c,r,color=colors[i%(len(colors))],label='obstacle')
            ax.scatter(co,ro,color=colors[(i+1)%(len(colors))],label='origin')
            ax.scatter(cs,rs,color=colors[(i+2)%(len(colors))],label='safe')

        ax.legend()

        # Save plot
        # plt.savefig('costmap.png')
        plt.show()


if __name__ == '__main__':
    rospy.init_node('Costmap_processor', anonymous=True)

    costmap_processor = ProcessCostmap()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo(f"[{rospy.get_name()}]" + "Shutting down")





           


