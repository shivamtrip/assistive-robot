import numpy as np
import heapq
from base_planner import BasePlanner


class AStarPlanner(BasePlanner):
    
    def __init__(self, start, goal, point_cloud):
        super().__init__(start, goal, point_cloud)
        
        self.resolutions = np.array(
            [
                0.01,  # x
                0.1,  # z
                1,  # phi
                0.1,  # ext
            ]
        ).tolist()
        
        
    
    def heuristic(self, current, goal):
        x, z, phi, ext = current
        tx, tz, tphi, text = goal
        dist = 0
        dist += abs(x - tx)
        dist += abs(z - tz)
        # dist += abs(phi - tphi)
        dist += abs(ext - text)
        return dist


    def convert_to_full_state(self, state):
        x, z, phi, ext = state
        full_state = np.array(self.start).copy()
        full_state[1] = x
        full_state[3] = z
        
        if phi == 1:
            full_state[4] = 0
        else:
            full_state[4] = np.pi
        full_state[5] = ext
        return full_state

    
    def generate_neighbors(self, current_state):
        new_state = np.array(current_state).copy()
        resolutions = self.resolutions
        neighbors = [
            new_state.copy() + np.array([resolutions[0], 0, 0, 0]),
            new_state.copy() - np.array([resolutions[0], 0, 0, 0]),

            new_state.copy() + np.array([0.0, resolutions[1], 0, 0]),
            new_state.copy() - np.array([0.0, resolutions[1], 0, 0]),

            new_state.copy() + np.array([0.0, 0.0, 0, resolutions[3]]),
            new_state.copy() - np.array([0.0, 0.0, 0, resolutions[3]]),
        ]

        new_state[2] *= -1
        neighbors.append(new_state.copy())

        finalneighbours = []
        
        for neighbor in neighbors:
            if self.check_valid_config(neighbor):
                finalneighbours.append(tuple(np.round(neighbor, 3)))

        return finalneighbours

    def config_to_workspace(self, state):
        new_state = np.array(state).copy()
        new_state = self.convert_to_full_state(new_state)
        return super().config_to_workspace(new_state)


    def convert_to_full_path(self, path):
        full_path = []
        for state in path:
            x, z, phi, ext = state
            full_state = self.convert_to_full_state(state)
            full_path.append(full_state)    
        return full_path

    def plan(self):
        path, found = self.astar()
        if found:
            path = self.convert_to_full_path(path)
        return path, found

    def astar(self):
        # x, y, theta, z, phi, ext
        # modify this state vector to -> (x, z, phi_ext (0 or 1), ext)
        start = self.start
        goal = self.goal
        orig_start = start.copy()
        start = np.array([
            start[0],
            start[3],
            start[4],
            start[5]
        ])
        goal = np.array([
            goal[0],
            goal[3],
            goal[4],
            goal[5]
        ])
        resolutions = self.resolutions
        start = tuple(start.tolist())
        goal = tuple(goal.tolist())
        open_list = []
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        heapq.heappush(open_list, (f_score[start], start))
        while open_list:
            _, current = heapq.heappop(open_list)
            current = tuple(np.round(np.array(current), 4))
            err = np.linalg.norm(np.array(self.config_to_workspace(current, orig_start)) - np.array(self.config_to_workspace(goal, orig_start)))
            print(err, current, len(open_list))
            if err < 0.1:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                print("Found Path. Path length = ", len(path))
                return path, True

            closed_set.add(current)
            neighbours = self.generate_neighbors(current)
            for neighbor in neighbours:
                if neighbor in closed_set:
                    continue
                
                g_score[neighbor] = g_score[current] + self.heuristic(current, neighbor)
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                
                list_of_opens = [x[1] for x in open_list]
                if neighbor in list_of_opens :
                    if g_score[neighbor] > g_score[current]:
                        continue
                came_from[neighbor] = current
                heapq.heappush(open_list, (f_score[neighbor], neighbor))
                    
        return [], False 