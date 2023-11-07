import numpy as np
import open3d as o3d 
import cv2
pcd = o3d.io.read_point_cloud("/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/sim/mesh.ply")

print('voxelization')
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.05)

voxels = voxel_grid.get_voxels()
indices = np.stack(list(vx.grid_index for vx in voxels))
colors = np.stack(list(vx.color for vx in voxels))
min_indices = np.min([vx.grid_index for vx in voxels], axis=0)
max_indices = np.max([vx.grid_index for vx in voxels], axis=0)

# Calculate the H, W, and D dimensions
height = max_indices[0] - min_indices[0] + 1
width = max_indices[1] - min_indices[1] + 1
depth = max_indices[2] - min_indices[2] + 1

hwd_array = np.full((height, width, depth, 3), [0, 0, 0], dtype=np.uint8)

# Fill the HWD array with voxel colors
for vx in voxels:
    grid_index = vx.grid_index - min_indices
    hwd_array[grid_index[0], grid_index[1], grid_index[2]] = vx.color
    
print(hwd_array.shape)

# from the reachability map, choose one configuration of the end effector that we wish to obtain. 
# From this, we plan from reverse. We go from the end effector configuration to the initial configuration.
# First, get the final configuration that we desire
# Once that is determined, we have to go to the "stow" location without any collision
# Stowing is done by tucking in the arm

# def convert_to_workspace(state):

def check_collision(state, voxel_grid):
    
    x, y, theta, z, phi, ext = state
    
    
    base_box = np.array([x, y, 0.15, ])
    arm_box = np.array([tx, ty, tz, w, h, d, theta])
    
    
    
    gripper_box = np.array([
        tx, 
        ty, 
        tz, 
        h, 
        w, 
        d, 
        theta
    ])
    boxes = [base_box, arm_box, gripper_box]    
    
    for box in boxes:
        newvoxels = np.zeros_like(voxel_grid)
        # fill newvoxels with box
        slice_img = newvoxels[:, :, 0].copy()
        box = box.astype(np.int) #TODO Convert to proper coordinate frame
        
        tx, ty, tz, w, h, d, theta = box
        nbox = cv2.cv.BoxPoints(np.array([tx, ty, w, h, theta]))
        nbox = np.int0(nbox)
        cv2.drawContours(slice_img, [nbox], 0, 255,-1)
        newvoxels[:, :, tz - d//2 : tz + d//2] = slice_img
        
        # check if there is any overlap
        if np.any(newvoxels & voxel_grid):
            return True
        
    return False
        
        
        
        
        
        
        


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
    
    
    