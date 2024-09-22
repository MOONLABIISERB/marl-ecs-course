import numpy as np
import matplotlib.pyplot as plt
import random

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
        self.agent_position = (1, 2) # Agent starts at (1, 2)
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
        color_grid = np.full((7, 6, 3), fill_value=[0.5, 0.5, 0.5])  # Default gray for the entire grid
        colors = {
            0: [0.8, 0.8, 0.8],  # Floor - light gray
            1: [0, 0, 0],        # Wall - black
            2: [0, 0, 1],        # Box - blue
            3: [0, 1, 0],        # Storage - green
            4: [1, 0, 0]         # Agent - red
        }
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                cell_value = self.grid[row][col]
                color_grid[row, col] = colors.get(cell_value, [0.5, 0.5, 0.5])
        agent_x, agent_y = self.agent_position
        color_grid[agent_x, agent_y] = colors[4]
        for box in self.box_positions:
            box_x, box_y = box
            color_grid[box_x, box_y] = colors[2]
        for storage in self.storage_positions:
            storage_x, storage_y = storage
            color_grid[storage_x, storage_y] = colors[3]
        plt.imshow(color_grid, interpolation='nearest')
        plt.xticks([]), plt.yticks([])
        plt.title("Sokoban Puzzle Environment")
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
            return self.agent_position  # If move is invalid, return the current position
        
    def is_valid_position(self, position):
        x, y = position
        if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
            return False
        if self.grid[x][y] == 1:  # Wall
            return False
        return True

    def step(self, action):
        if self.done:
            print("Episode is done. Reset the environment.")
            return self.agent_position, 0, self.done  # Ensure it returns values even if done

        # Move the agent
        previous_agent_position = self.agent_position
        self.agent_position = self.get_next_state(self.actions[action])

        # Check for rewards and termination conditions
        if self.agent_position != previous_agent_position:
            # If a box was pushed to a storage location
            if previous_agent_position in self.box_positions:
                box_index = self.box_positions.index(previous_agent_position)
                if self.box_positions[box_index] in self.storage_positions:
                    reward = 0  # Box is at storage
                else:
                    reward = -1  # Box is not at storage
            else:
                reward = -1  # General step cost

            # Check if all boxes are on storage locations
            if all(box in self.storage_positions for box in self.box_positions):
                print("All boxes are placed in storage! Episode complete.")
                self.done = True

            # Check if a box gets stuck (simple check for corners)
            for box in self.box_positions:
                if self.is_box_stuck(box):
                    print("A box is stuck! Episode complete.")
                    self.done = True
        else:
            # Add penalty for hitting a wall
            if not self.is_valid_position(self.agent_position):
                reward = -10  # Penalty for hitting a wall
            else:
                reward = -1  # No movement means penalty

        return self.agent_position, reward, self.done  # Ensure this is always returned

        return self.agent_position, reward, self.done

    def is_box_stuck(self, box_position):
        x, y = box_position
        if (self.grid[x-1][y] == 1 and self.grid[x][y-1] == 1) or \
           (self.grid[x-1][y] == 1 and self.grid[x][y+1] == 1) or \
           (self.grid[x+1][y] == 1 and self.grid[x][y-1] == 1) or \
           (self.grid[x+1][y] == 1 and self.grid[x][y+1] == 1):
            return True
        return False


if __name__ == "__main__":
    env = SokobanEnvironment()
    env.display_grid()