import numpy as np
from planner.base_planner import BasePlanner



class NaivePlanner(BasePlanner):
    
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

    def plan(self):
        tx, ty, ttheta, tz, tphi, text = self.goal
        path = []
        next_state = self.start.copy()
        next_state[:2] = np.array([tx, ty])
        if self.check_valid_config(next_state):
            path.append(next_state.copy())
        else:
            return path, False
        
        next_state[2] = ttheta
        if self.check_valid_config(next_state):
            path.append(next_state.copy())
        else:
            return path, False
        
        next_state[3] = tz
        if self.check_valid_config(next_state):
            path.append(next_state.copy())
        else:
            return path, False
        
        next_state[4] = tphi 
        if self.check_valid_config(next_state):
            path.append(next_state.copy())
        else:
            return path, False
        
        next_state[5] = text
        if self.check_valid_config(next_state):
            path.append(next_state.copy())
        else:
            return path, False
        
            
        delay = 0.1
        plan = np.array(path)
        
        # interp_waypoints = [plan[0]]
        
        # for i in range(1, len(plan)):
        #     for j in range(int(1/delay)):
        #         delta = (plan[i] - plan[i-1]) * j * delay
        #         interp_waypoints.append(plan[i-1] + delta)

        return plan, True