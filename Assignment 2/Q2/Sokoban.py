import numpy as np
from enum import Enum

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SokobanEnv:
    def __init__(self):
        self.height = 6
        self.width = 7
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.grid[0, :] = self.grid[-1, :] = 1  # walls
        self.grid[:, 0] = self.grid[:, -1] = 1  # walls
        self.agent_pos = [1, 1]
        self.box_pos = [2, 3]
        self.goal_pos = [4, 5]
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        new_agent_pos = self.agent_pos.copy()
        if action == Action.UP:
            new_agent_pos[0] -= 1
        elif action == Action.RIGHT:
            new_agent_pos[1] += 1
        elif action == Action.DOWN:
            new_agent_pos[0] += 1
        elif action == Action.LEFT:
            new_agent_pos[1] -= 1

        if self._is_valid_move(new_agent_pos):
            if new_agent_pos == self.box_pos:
                new_box_pos = [2 * new_agent_pos[0] - self.agent_pos[0],
                               2 * new_agent_pos[1] - self.agent_pos[1]]
                if self._is_valid_move(new_box_pos):
                    self.box_pos = new_box_pos
                    self.agent_pos = new_agent_pos
                    if self.box_pos == self.goal_pos:
                        self.done = True
                        return self._get_state(), 0, True, {}
                else:
                    return self._get_state(), -1, False, {}
            else:
                self.agent_pos = new_agent_pos

        return self._get_state(), -1, False, {}

    def _is_valid_move(self, pos):
        return 0 <= pos[0] < self.height and 0 <= pos[1] < self.width and self.grid[pos[0], pos[1]] == 0

    def _get_state(self):
        return tuple(self.agent_pos + self.box_pos)

    def render(self):
        render_grid = self.grid.copy()
        render_grid[self.agent_pos[0], self.agent_pos[1]] = 2
        render_grid[self.box_pos[0], self.box_pos[1]] = 3
        render_grid[self.goal_pos[0], self.goal_pos[1]] = 4
        print(render_grid)

def get_all_states(env):
    states = []
    for ax in range(1, env.height - 1):
        for ay in range(1, env.width - 1):
            for bx in range(1, env.height - 1):
                for by in range(1, env.width - 1):
                    if [ax, ay] != [bx, by]:
                        states.append((ax, ay, bx, by))
    return states

def get_transition_prob(state, action, next_state, env):
    temp_env = SokobanEnv()
    temp_env.agent_pos = [state[0], state[1]]
    temp_env.box_pos = [state[2], state[3]]
    temp_env.done = False
    
    new_state, _, _, _ = temp_env.step(action)
    
    if new_state == next_state:
        return 1.0
    return 0.0

def get_reward(state, action, next_state, env):
    temp_env = SokobanEnv()
    temp_env.agent_pos = [state[0], state[1]]
    temp_env.box_pos = [state[2], state[3]]
    temp_env.done = False
    
    _, reward, _, _ = temp_env.step(action)
    return reward