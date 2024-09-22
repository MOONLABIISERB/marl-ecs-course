import numpy as np
import gym 
from gym import spaces

class SokobanEnv(gym.Env):
    def __init__(self):
        # grid dimensions  6x7
        self.grid_size = (6, 7)  # (Rows, Columns)

        self.action_space = spaces.Discrete(4)  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT

        # Observation space: agent and boxes' positions on the grid
        self.observation_space = spaces.Box(low=0, high=1, shape=(6, 7, 3), dtype=np.int32)
        # 3 channels: Agent position, Box positions, Storage locations
        
        # grid with walls (1), floor (0), boxes (2), storage (3)
        self.grid = np.zeros((6, 7))  #  6x7 grid (walls are 1, floor 0)
        
        # Adding walls around the grid
        self.grid[:, 0] = self.grid[:, -1] = self.grid[0, :] = self.grid[-1, :] = 1
        
        # Defining initial positions for agent, boxes, and storage
        self.agent_position = [1, 1]  # Agent starts at row 1, col 1
        self.boxes = [[2, 2], [4, 5]]  # Two boxes at specific locations
        self.storage_locations = [[3, 3], [5, 5]]  # Storage locations
        
        self.reset()

    def reset(self):
        """Reset the environment."""
        self.grid = np.zeros((6, 7))
        self.grid[:, 0] = self.grid[:, -1] = self.grid[0, :] = self.grid[-1, :] = 1  # Walls
        self.agent_position = [1, 1]  # Reset agent's position
        self.boxes = [[2, 2], [4, 5]]  # Reset boxes
        self.steps = 0
        self.done = False
        return self.get_observation()

    def get_observation(self):
        """Return the observation (current state of the environment)."""
        obs = np.zeros((6, 7, 3))
        obs[self.agent_position[0], self.agent_position[1], 0] = 1  # Agent channel
        for box in self.boxes:
            obs[box[0], box[1], 1] = 1  # Box channel
        for storage in self.storage_locations:
            obs[storage[0], storage[1], 2] = 1  # Storage channel
        return obs

    def step(self, action):
        """Take a step in the environment."""
        self.steps += 1

        reward = -1 
        next_pos = self.agent_position.copy()

        # Move agent based on action
        if action == 0:  # UP
            next_pos[0] -= 1
        elif action == 1:  # DOWN
            next_pos[0] += 1
        elif action == 2:  # LEFT
            next_pos[1] -= 1
        elif action == 3:  # RIGHT
            next_pos[1] += 1
        if self.grid[next_pos[0], next_pos[1]] == 1:
            reward -= 10  
        else:
            self.agent_position = next_pos
        
        for i, box in enumerate(self.boxes):
            if self.agent_position == box:
                next_box_pos = [box[0] + (next_pos[0] - box[0]), box[1] + (next_pos[1] - box[1])]
                
                if self.grid[next_box_pos[0], next_box_pos[1]] == 1 or next_box_pos in self.boxes:
                    reward -= 20  
                else:
                    self.boxes[i] = next_box_pos
        
        if all(box in self.storage_locations for box in self.boxes):
            reward = 1 
            self.done = True
        
        return self.get_observation(), reward, self.done, {}

    def render(self):
        """Render the current state of the environment."""
        print("Agent:", self.agent_position)
        print("Boxes:", self.boxes)
        print(self.grid)


env = SokobanEnv()

obs = env.reset()
env.render()

done = False
while not done:
    action = env.action_space.sample() 
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"Reward: {reward}, Done: {done}")
