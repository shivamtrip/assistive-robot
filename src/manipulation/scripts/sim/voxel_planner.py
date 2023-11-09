import numpy as np
import open3d as o3d 
import cv2
import heapq
og_pcd = o3d.io.read_point_cloud("/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/sim/mesh.ply")

pcd_min = np.min(np.array(og_pcd.points), axis=0)
resolution = 0.05

l, l1, l3 = 0.11587, 0.25, 0.25
def get_rotation_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]], dtype = np.float64)
    

def get_boxes(state, get_centers = False):
    boxes = []
    x, y, theta, z, phi, ext = state
    centr = np.array([
        x - l * np.cos(theta), 
        y - l * np.sin(theta), 0.093621], dtype = np.float64).reshape(3, 1)
    base_box_o3d = o3d.geometry.OrientedBoundingBox(
        center = centr,
        R = get_rotation_mat(theta).reshape(3, 3),
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
        R = get_rotation_mat(theta).reshape(3, 3),
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
        R = get_rotation_mat(theta + phi).reshape(3, 3),
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
    
def check_collision(state, pcd):
    boxes, centers = get_boxes(state, get_centers = False)
    points_in_pcd = o3d.utility.Vector3dVector(np.array(pcd.points))
    for box in boxes:
        if len(box.get_point_indices_within_bounding_box(points_in_pcd)) > 0:
            return True
    return False
        

check_collision(np.array([0, 0, np.pi/2, 0.5, np.pi/3, 0.5]), og_pcd)

def check_valid_config(state):
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
    return True
    


def plan_naive_path(curstate, target):
    tx, ty, ttheta, tz, tphi, text = target
    path = []
    next_state = curstate.copy()
    next_state[:2] = np.array([tx, ty])
    if not check_collision(next_state, og_pcd):
        path.append(next_state.copy())
    else:
        return path, False
    
    next_state[2] = ttheta
    if not check_collision(next_state, og_pcd):
        path.append(next_state.copy())
    else:
        return path, False
    
    next_state[3] = tz
    if not check_collision(next_state, og_pcd):
        path.append(next_state.copy())
    else:
        return path, False
    
    next_state[4] = tphi 
    if not check_collision(next_state, og_pcd):
        path.append(next_state.copy())
    else:
        return path, False
    
    next_state[5] = text
    if not check_collision(next_state, og_pcd):
        path.append(next_state.copy())
    else:
        return path, False
    print(next_state.shape)
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
    
def config_to_workspace(state):
    x, y, theta, z, phi, ext = state
    ee_x = x - l * np.cos(theta) + (0.3 + ext - l1/2) * np.sin(theta) + l3 * np.sin(theta + phi)
    ee_y = y - l * np.sin(theta) - (0.3 + ext - l1/2) * np.cos(theta) - l3 * np.cos(theta + phi)
    ee_z = z - 0.15    
    return ee_x, ee_y, ee_z
def circular_distance_heuristic(angle1, angle2):
    diff = abs(angle1 - angle2) % (2 * np.pi)
    return min(diff, 2 * np.pi - diff)

def heuristic(current, goal):
    # angle distance 
    
    theta_dist = circular_distance_heuristic(current[2], goal[2])
    theta_dist += circular_distance_heuristic(current[4], goal[4])
    # x, y, theta, z, phi, ext
    
    cartesian_distance = np.linalg.norm(np.array(config_to_workspace(current)) - np.array(config_to_workspace(goal)))
    norm_dist = np.linalg.norm(
        np.array(
            [
                current[0], current[1], current[3], current[4], current[5]
            ]
        ) - np.array(
            [
                goal[0], goal[1], goal[3], goal[4], goal[5]
            ]
        )
        
    )
    
    

    return norm_dist

# Define a function to generate neighboring states
def generate_neighbors(current_state, resolutions):
    neighbors = []
    for i in range(6):
        for delta in [-1, 1]:
            new_state = np.array(current_state).copy()
            new_state[i] += delta * resolutions[i]
            # new_state
            # round the state
            # new_state = np.round(new_state, 3)
            # print(new_state)
            new_state =  np.round(new_state, 4)
            if check_valid_config(new_state): #& (not check_collision(new_state, og_pcd)):
                neighbors.append(tuple(new_state))
        #     print("Valid")
        # else:
        #     print("Invalid")
    
    neighbors.append(tuple(current_state))
    return neighbors


def astar(start, goal):
    resolutions = np.array(
        [
            0.01,  # x
            0.01,  # y
            0.1,  # theta
            0.1,  # z
            0.1,  # phi
            0.05,  # ext
        ]
    ).tolist()
    start = tuple(start.tolist())
    goal = tuple(goal.tolist())
    open_list = []
    closed_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    heapq.heappush(open_list, (f_score[start], start))

    while open_list:
        _, current = heapq.heappop(open_list)
        current = tuple(np.round(np.array(current), 4))
        err = np.linalg.norm(np.array(config_to_workspace(current)) - np.array(config_to_workspace(goal)))
        print(err, current)
        if err < 0.5:
        
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current.tolist()]
            path.append(start)
            path.reverse()
            return path

        closed_set.add(current)
        neighbours = generate_neighbors(current, resolutions)
        # print(len(neighbours))
        for neighbor in neighbours:
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in open_list or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))
    print(open_list)
    return None 


def dfs(initial_state, goal_state, resolutions, ):
    stack = [initial_state]
    path = {tuple(initial_state): None}

    while stack:
        current_state = stack.pop()
        
        err = np.linalg.norm(np.array(config_to_workspace(current_state)) - np.array(config_to_workspace(goal_state)))
        print(err, current_state)
        if err < 0.5:
            # Reconstruct the path
            result = []
            while current_state is not None:
                result.insert(0, current_state)
                current_state = path.get(tuple(current_state))
            return result

        for dim in range(6):
            for delta in [-resolutions[dim], resolutions[dim]]:
                new_state = list(current_state)
                new_state[dim] += delta

                new_state = np.round(new_state, 4)
                if check_valid_config(new_state) and tuple(new_state) not in path and not check_collision(new_state, og_pcd):
                    stack.append(new_state)
                    path[tuple(new_state)] = current_state

    return None

def visualize(state):
    boxes, centers =  get_boxes(state, get_centers = True)
    o3d.visualization.draw_geometries([og_pcd] + boxes + centers)

if __name__ == "__main__":
    # x, y, theta, z, phi, ext
    start =  np.array([  0,   0,       0, 0.3, 3.1,   0])
    target = np.array([0.0, 0.1,  1.57, 1.3,     0, 0.1])
    # path = astar(start, target)
    resolutions = np.array(
        [
            0.01,  # x
            0.01,  # y
            0.1,  # theta
            0.1,  # z
            0.1,  # phi
            0.05,  # ext
        ]
    ).tolist()
    # path = dfs(start, target, resolutions)
    path, successs = plan_naive_path(start, target)
    if path is not None :
        print("Found path:", len(path))
    else:
        print("Path not found")
    # visualize(target)
    	
    
    for i in range(0, len(path), len(path)//10 + 1):
        print(path[i])
        visualize(path[i])
        
    boxes, centers =  get_boxes(path[-1], get_centers = True)
    o3d.visualization.draw_geometries([og_pcd] + boxes + centers)
    # boxes, centers =  get_boxes(path[i], get_centers = True)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # boxes, centers =  get_boxes(path[i], get_centers = True)
    # for box in boxes:
    #     vis.add_geometry(box)
    # for center in centers:
    #     vis.add_geometry(center)
    # vis.add_geometry(og_pcd)
    # for state in path:
    #     boxes, centers =  get_boxes(path[i], get_centers = True)
    #     for box in boxes:
    #         vis.update_geometry(box)
    #     for center in centers:
    #         vis.update_geometry(center)

    #     vis.poll_events()
    #     vis.update_renderer()
