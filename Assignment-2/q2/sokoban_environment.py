import numpy as np
import matplotlib.pyplot as plt

class SokobanEnvironment:
    def __init__(self):
        self.grid = [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 1],
            [1, 3, 0, 0, 0, 1],
            [1, 0, 0, 2, 0, 1],
            [1, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]
        self.agent_position = (1, 2) 
        self.box_positions = [(4, 3)]
        self.storage_positions = [(3, 1)]
        self.actions = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }
        self.done = False

    def display_grid(self):
        fig, ax = plt.subplots()
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.grid[row][col] == 1:
                    ax.add_patch(plt.Rectangle((col, row), 1, 1, edgecolor='black', facecolor='black')) 
                elif (row, col) == self.agent_position:
                    ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, edgecolor='red', facecolor='white'))  
                elif (row, col) in self.box_positions:
                    ax.add_patch(plt.Rectangle((col, row), 1, 1, edgecolor='blue', facecolor='yellow')) 
                elif (row, col) in self.storage_positions:
                    ax.plot(col + 0.5, row + 0.5, marker='x', color='green', markersize=20, markeredgewidth=3)  
                else:
                    ax.add_patch(plt.Rectangle((col, row), 1, 1, edgecolor='gray', facecolor='gray'))  
        ax.set_xlim(0, len(self.grid[0]))
        ax.set_ylim(0, len(self.grid))
        ax.set_aspect('equal')
        plt.gca().invert_yaxis() 
        plt.show()

    def reset(self):
        self.agent_position = (1, 2)
        self.box_positions = [(4, 3)]
        self.done = False

    def get_next_state(self, action):
        dx, dy = action
        x, y = self.agent_position
        next_agent_position = (x + dx, y + dy)

        if self.is_valid_position(next_agent_position):
            if next_agent_position in self.box_positions:
                next_box_position = (next_agent_position[0] + dx, next_agent_position[1] + dy)
                if self.is_valid_position(next_box_position) and next_box_position not in self.box_positions:
                    self.box_positions.remove(next_agent_position)
                    self.box_positions.append(next_box_position)
                    return next_agent_position
                else:
                    return self.agent_position
            else:
                return next_agent_position
        else:
            return self.agent_position  
        
    def is_valid_position(self, position):
        x, y = position
        if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
            return False
        if self.grid[x][y] == 1:  
            return False
        return True

    def step(self, action):
        if self.done:
            return self.agent_position, 0, self.done 

        
        previous_agent_position = self.agent_position
        self.agent_position = self.get_next_state(self.actions[action])

        
        if self.agent_position != previous_agent_position:
           
            if previous_agent_position in self.box_positions:
                box_index = self.box_positions.index(previous_agent_position)
                if self.box_positions[box_index] in self.storage_positions:
                    reward = 5 
                else:
                    reward = -1 
            else:
                reward = -1  

            if all(box in self.storage_positions for box in self.box_positions):
                print("All boxes are placed in storage! Episode complete.")
                self.done = True

            for box in self.box_positions:
                if self.is_box_stuck(box):
                    print("A box is stuck! Episode complete.")
                    self.done = True
        else:
            if not self.is_valid_position(self.agent_position):
                reward = -10  
            else:
                reward = -1 

        return self.agent_position, reward, self.done 

    def is_box_stuck(self, box_position):
        x, y = box_position
        if (self.grid[x-1][y] == 1 and self.grid[x][y-1] == 1) or \
           (self.grid[x-1][y] == 1 and self.grid[x][y+1] == 1) or \
           (self.grid[x+1][y] == 1 and self.grid[x][y-1] == 1) or \
           (self.grid[x+1][y] == 1 and self.grid[x][y+1] == 1):
            return True
        return False
