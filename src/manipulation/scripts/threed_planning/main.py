import rospy
import cv2
import numpy as np


def parse_scene():
    """
    here, we scan the environment using just head, get RGBD/point clouds, and fuse them to form a cohesive 
    point cloud of the scene. This point cloud is then passed to the planner. 
    """
    pass

def voxelize_scene():
    """
    here, we voxelize the scene to get a 3D grid representation of the scene. This is then passed to the planner. 
    """
    pass

def convert_config_to_ee(config_point):
    r = 0.12
    h = 0.05
    x, y, z, theta = config_point
    x_hat = x + r * np.sin(theta)
    y_hat = y + r * np.cos(theta)
    z_hat = z - h
    
    return np.array([x_hat, y_hat, z_hat])
    

def voxelize_config_space(ee_voxels):
    """
    here, we convert the voxels obtained from end effector space to config space. This is then passed to the planner. 
    """
    r = 0.12
    xmin, xmax  = -r, r
    ymin, ymax = 0.0, 1.0
    zmin, zmax = 0.0, 1.05
    theta_min, theta_max = -np.pi, np.pi
    
    xres, yres, zres, theta_res = 0.05, 0.05, 0.05, 0.05
    
    x = np.arange(xmin, xmax, xres)
    y = np.arange(ymin, ymax, yres)
    z = np.arange(zmin, zmax, zres)
    theta = np.arange(theta_min, theta_max, theta_res)
    
    
    i, j, k, l = np.indices((len(x), len(y), len(z), len(theta)))
    config_points = np.stack((x[i], y[j], z[k], theta[l]), axis=-1)
    ee_points = convert_config_to_ee(config_points)
    ee_indices = ee_points.astype(int)
    config_voxels = ee_voxels[ee_indices[..., 0], ee_indices[..., 1], ee_indices[..., 2], ee_indices[..., 3]]
    
    
    
    
    
    

# we have the voxel representation of the scene. lets now perform end-effector planning

def plan_end_effector(start_point, target_point, voxels):
    """
    here, we plan the end-effector trajectory to reach the target point. 
    start_point = (yaw, x, y, z) in scene frame.`
    target_point = (yaw, x, y, z) in scene frame.
    voxels = 3D grid representation of the scene with each cell being probability of occupancy. (N, M, K)
    """
    yaw, x, y, z = target_point
    
    # https://github.com/hello-robot/stretch_tutorials/blob/master/stretch_body/jupyter/inverse_kinematics.ipynb