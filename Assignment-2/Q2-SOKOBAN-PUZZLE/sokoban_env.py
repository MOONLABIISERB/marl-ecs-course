# sokoban_env.py

import numpy as np
import gym
from gym import spaces


class SokobanEnv(gym.Env):
    """
    Custom Sokoban environment for reinforcement learning.
    """

    def __init__(self):
        super(SokobanEnv, self).__init__()

        # Define the grid size
        self.grid_height = 6
        self.grid_width = 7

        # Define action and observation space
        # Actions: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)
        # Observation: The grid representation
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(self.grid_height, self.grid_width), dtype=np.uint8
        )

        # Define grid elements
        self.EMPTY = 0
        self.WALL = 1
        self.BOX = 2
        self.TARGET = 3
        self.PLAYER = 4
        self.BOX_ON_TARGET = 5

        # Map actions to movements
        self.action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }

        self.reset()

    def reset(self):
        # Initialize the grid
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        self.grid[1:-1, 1:-1] = self.EMPTY  # Set inner area to empty

        # Add walls around the grid
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

        # Place the player at a random empty position
        empty_positions = list(zip(*np.where(self.grid == self.EMPTY)))
        self.player_position = np.array(empty_positions[np.random.choice(len(empty_positions))])
        self.grid[self.player_position[0], self.player_position[1]] = self.PLAYER

        # Place the box at a random empty position
        empty_positions = list(zip(*np.where(self.grid == self.EMPTY)))
        self.box_position = np.array(empty_positions[np.random.choice(len(empty_positions))])
        self.grid[self.box_position[0], self.box_position[1]] = self.BOX

        # Place the target at a random empty position
        empty_positions = list(zip(*np.where(self.grid == self.EMPTY)))
        self.target_position = np.array(empty_positions[np.random.choice(len(empty_positions))])
        self.grid[self.target_position[0], self.target_position[1]] = self.TARGET

        return self.grid.copy(), {}

    def step(self, action):
        move = self.action_map[action]
        new_player_pos = self.player_position + move

        # Check for wall collision
        if self.grid[new_player_pos[0], new_player_pos[1]] == self.WALL:
            return self.grid.copy(), -1, False, False, {}

        # Check if the player is pushing the box
        if np.array_equal(new_player_pos, self.box_position):
            new_box_pos = self.box_position + move

            # Check if the box can be moved
            if self.grid[new_box_pos[0], new_box_pos[1]] in [self.WALL, self.BOX]:
                return self.grid.copy(), -1, False, False, {}

            # Move the box
            self.grid[self.box_position[0], self.box_position[1]] = self.EMPTY
            self.box_position = new_box_pos

            if np.array_equal(self.box_position, self.target_position):
                self.grid[self.box_position[0], self.box_position[1]] = self.BOX_ON_TARGET
            else:
                self.grid[self.box_position[0], self.box_position[1]] = self.BOX

        # Move the player
        self.grid[self.player_position[0], self.player_position[1]] = self.EMPTY
        self.player_position = new_player_pos
        self.grid[self.player_position[0], self.player_position[1]] = self.PLAYER

        # Check if the box is on the target
        done = np.array_equal(self.box_position, self.target_position)
        reward = 10 if done else -1  # Positive reward if solved, negative otherwise

        return self.grid.copy(), reward, done, False, {}

    def render(self, mode="human"):
        if mode == "human":
            symbols = {
                self.EMPTY: ' ',
                self.WALL: '#',
                self.BOX: '$',
                self.TARGET: '.',
                self.PLAYER: '@',
                self.BOX_ON_TARGET: '*'
            }
            print("\n".join("".join(symbols[cell] for cell in row) for row in self.grid))
        elif mode == "rgb_array":
            # Optional implementation for visualizing the environment
            pass

    def get_state(self):
        # Returns a tuple representing the current state
        return (tuple(self.player_position), tuple(self.box_position), tuple(self.target_position))
