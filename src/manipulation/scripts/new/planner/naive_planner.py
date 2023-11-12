import numpy as np
from base_planner import BasePlanner



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
        if not self.check_valid_config(next_state):
            path.append(next_state.copy())
        else:
            return path, False
        
        next_state[2] = ttheta
        if not self.check_valid_config(next_state):
            path.append(next_state.copy())
        else:
            return path, False
        
        next_state[3] = tz
        if not self.check_valid_config(next_state):
            path.append(next_state.copy())
        else:
            return path, False
        
        next_state[4] = tphi 
        if not self.check_valid_config(next_state):
            path.append(next_state.copy())
        else:
            return path, False
        
        next_state[5] = text
        if not self.check_valid_config(next_state):
            path.append(next_state.copy())
        else:
            return path, False

        return path, True