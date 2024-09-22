import numpy as np
import time
import os

class SokobanEnv:
    def __init__(self):
        # Initialize grid based on the provided layout
        self.grid = np.array([
            ['#', '#', '#', '#', '#', '#'],
            ['#', ' ', 'A', '#', '#', '#'],
            ['#', ' ', ' ', '#', '#', '#'],
            ['#', 'G', ' ', ' ', ' ', '#'],
            ['#', ' ', ' ', 'B', ' ', '#'],
            ['#', ' ', ' ', '#', '#', '#'],
            ['#', '#', '#', '#', '#', '#']
        ])
        self.agent_pos = [1, 2]  # Starting position of the agent (1, 2)
        self.box_pos = [4, 3]    # Starting position of the box (4, 3)
        self.goal_pos = [3, 1]   # Position of the goal (3, 1)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
    
    def reset(self):
        self.agent_pos = [1, 2]
        self.box_pos = [4, 3]
        return self.grid_state()

    def grid_state(self):
        # Create a copy of the grid to update agent and box positions
        grid_copy = self.grid.copy()
        grid_copy[self.agent_pos[0], self.agent_pos[1]] = 'A'  # Place the agent
        grid_copy[self.box_pos[0], self.box_pos[1]] = 'B'  # Place the box
        return grid_copy

    def step(self, action):
        next_agent_pos = self.agent_pos[:]
        if action == 'UP':
            next_agent_pos[0] -= 1
        elif action == 'DOWN':
            next_agent_pos[0] += 1
        elif action == 'LEFT':
            next_agent_pos[1] -= 1
        elif action == 'RIGHT':
            next_agent_pos[1] += 1
        
        # Check if the agent hits a wall
        if self.grid[next_agent_pos[0], next_agent_pos[1]] == '#':
            return self.grid_state(), -1, False  # Invalid move, penalty
        
        # Check if the agent pushes the box
        if next_agent_pos == self.box_pos:
            next_box_pos = self.box_pos[:]
            if action == 'UP':
                next_box_pos[0] -= 1
            elif action == 'DOWN':
                next_box_pos[0] += 1
            elif action == 'LEFT':
                next_box_pos[1] -= 1
            elif action == 'RIGHT':
                next_box_pos[1] += 1
            
            # Check if the box hits a wall or another box
            if self.grid[next_box_pos[0], next_box_pos[1]] == '#':
                return self.grid_state(), -1, False  # Invalid move, penalty
            
            # Move the box
            self.box_pos = next_box_pos
        
        # Move the agent
        self.agent_pos = next_agent_pos
        
        # Check if the box is at the goal
        if self.box_pos == self.goal_pos:
            return self.grid_state(), 0, True  # Success, puzzle solved
        
        return self.grid_state(), -1, False  # Normal step, no goal reached
    
    def render(self):
        os.system('clear')  # For Linux/OS X (use 'cls' for Windows)
        print("\n".join(["".join(row) for row in self.grid_state()]))
        time.sleep(0.5)

# Test Environment Visualization
env = SokobanEnv()
env.render()

# Simulate some moves
actions = ['LEFT', 'LEFT', 'DOWN', 'RIGHT', 'RIGHT']
for action in actions:
    state, reward, done = env.step(action)
    env.render()
    if done:
        print("Puzzle Solved!")
        break
