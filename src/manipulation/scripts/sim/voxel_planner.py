import numpy as np
import open3d as o3d 
import cv2
og_pcd = o3d.io.read_point_cloud("/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/sim/mesh.ply")

pcd_min = np.min(np.array(og_pcd.points), axis=0)
resolution = 0.05

print('voxelization')
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(og_pcd,
                                                            voxel_size=resolution)

voxels = voxel_grid.get_voxels()
indices = np.stack(list(vx.grid_index for vx in voxels))
colors = np.stack(list(vx.color for vx in voxels))
min_indices = np.min([vx.grid_index for vx in voxels], axis=0)
max_indices = np.max([vx.grid_index for vx in voxels], axis=0)

# Calculate the H, W, and D dimensions
height = max_indices[0] - min_indices[0] + 1
width = max_indices[1] - min_indices[1] + 1
depth = max_indices[2] - min_indices[2] + 1

hwd_array = np.full((height, width, depth), 0, dtype=np.uint8)

# Fill the HWD array with voxel colors
for vx in voxels:
    grid_index = vx.grid_index - min_indices
    hwd_array[grid_index[0], grid_index[1], grid_index[2]] = 1


def voxel_to_pcd(voxel):
    # voxel is a 3D array
    indices = np.where(voxel != 0)
    
    points = np.stack([indices[0], indices[1], indices[2]], axis=1)
    points = points.astype(np.float32)
    points[:, 0] = points[:, 0] * resolution 
    points[:, 1] = points[:, 1] * resolution 
    points[:, 2] = points[:, 2] * resolution 
    
    points += np.array(pcd_min)
    
    return points




# from the reachability map, choose one configuration of the end effector that we wish to obtain. 
# From this, we plan from reverse. We go from the end effector configuration to the initial configuration.
# First, get the final configuration that we desire
# Once that is determined, we have to go to the "stow" location without any collision
# Stowing is done by tucking in the arm

# def convert_to_workspace(state):

    
def get_rotation_mat(theta):
    # theta = -theta
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]], dtype = np.float64)
    
    

def check_collision(state, voxel_grid, resolution, xmin, ymin, zmin):
    
    x, y, theta, z, phi, ext = state
    pcds = [
        og_pcd,
        
        # arm_box_o3d,
        # gripper_box_o3d
    ]
    l = 0.11587
    centr = np.array([
        x - l * np.cos(theta), 
        y - l * np.sin(theta), 0.093621], dtype = np.float64).reshape(3, 1)
    base_box_o3d = o3d.geometry.OrientedBoundingBox(
        center = centr,
        R = get_rotation_mat(theta).reshape(3, 3),
        extent = np.array([0.35, 0.35, 0.3], dtype = np.float64).reshape(3, 1)
    )
    center = o3d.geometry.TriangleMesh.create_sphere(radius=0.03).compute_vertex_normals()
    center.translate(np.array([x, y, 0.09]))
    center.paint_uniform_color([0, 0, 1])
    pcds.append(center)
    pcds.append(base_box_o3d)
    center = o3d.geometry.TriangleMesh.create_sphere(radius=0.03).compute_vertex_normals()
    center.translate(centr)
    center.paint_uniform_color([0, 0, 1])
    pcds.append(center)
    l1 = 0.25
    arm_center = centr + np.array(
            [
                (- l1 * np.sin(theta) + ( 0.3 + ext)* np.sin(theta))/2,
                (l1 * np.cos(theta)  - ( 0.3 + ext) * np.cos(theta))/2,
                z
            ]
        ).reshape(3, 1)
    arm_box_o3d = o3d.geometry.OrientedBoundingBox(
        center = arm_center,
        R = get_rotation_mat(theta).reshape(3, 3),
        extent = np.array([0.1, ext + 0.3, 0.1], dtype = np.float64).reshape(3, 1)
    )
    center = o3d.geometry.TriangleMesh.create_sphere(radius=0.03).compute_vertex_normals()
    center.translate(arm_center)
    center.paint_uniform_color([0, 0, 1])
    pcds.append(center)
    pcds.append(arm_box_o3d)
    l3 = 0.25
    gripper_center = centr + np.array([
        (0.3 + ext - l1/2) * np.sin(theta) + l3/2 * np.sin(theta + phi),
        -(0.3 + ext - l1/2) * np.cos(theta) - l3/2 * np.cos(theta + phi),
        z - 0.15
    ]).reshape(3, 1)
    
    gripper_box_o3d = o3d.geometry.OrientedBoundingBox(
        center = gripper_center,
        R = get_rotation_mat(theta + phi).reshape(3, 3),
        extent = np.array([0.15, 0.25, 0.15], dtype = np.float64).reshape(3, 1)
    )
    
    center = o3d.geometry.TriangleMesh.create_sphere(radius=0.03).compute_vertex_normals()
    center.translate(gripper_center)
    center.paint_uniform_color([0, 0, 1])
    pcds.append(center)
    pcds.append(gripper_box_o3d)
    
    
    
    o3d.visualization.draw_geometries(pcds)

    return False
        
# points = voxel_to_pcd(hwd_array)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd])
# x, y, theta, z, phi, ext
check_collision(np.array([0, 0, np.pi/2, 0.5, np.pi/3, 0.5]), hwd_array, resolution, pcd_min[0], pcd_min[1], pcd_min[2])
        


def plan_naive_path(target):
    tx, ty, ttheta, tz, tphi, text = target
    path = []
    next_state = np.array([tx, ty, 0, 0, 0, 0])
    if not check_collision(next_state):
        path.append(next_state)
    else:
        return path, False
    
    next_state = np.array([tx, ty, ttheta, 0, 0, 0])
    if not check_collision(next_state):
        path.append(next_state)
    else:
        return path, False
    
    next_state = np.array([tx, ty, ttheta, tz, 0, 0])
    if not check_collision(next_state):
        path.append(next_state)
    else:
        return path, False
    
    next_state = np.array([tx, ty, ttheta, tz, tphi, 0])
    if not check_collision(next_state):
        path.append(next_state)
    else:
        return path, False
    
    next_state = np.array([tx, ty, ttheta, tz, tphi, text])
    if not check_collision(next_state):
        path.append(next_state)
    else:
        return path, False
    
    return path, True
    

def rrt(target):
    # state is x, y, theta, z, phi, text
    state = np.array([0, 0, 0, 0, 0, 0])
    
    # we choose actions discretely. Actions are as follows
    # rotate by theta
    # translate by x
    # pull lift by z
    # rotate arm by phi
    # extend arm by y
    
    movement_map_max = {
        0 : 45 * np.pi/180,
        1 : 0.3, 
        2 : 0.5,
        3 : 45 * np.pi/180,
        4 : 0.3
    }
    for i in range(1000):
        random_action = np.random.randint(0, 5)
        dist_to_move = np.random.uniform(0, movement_map_max[random_action])
        state[random_action] += movement_map_max[random_action] * np.random.choice([-1, 1])
    
    
    