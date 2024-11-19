import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SokobanEnv(gym.Env):
    def __init__(self):
        super(SokobanEnv, self).__init__()

        self.height = 6
        self.width = 7

        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(self.height, self.width), dtype=np.uint8
        )

        self.WALL = 0
        self.FLOOR = 1
        self.BOX = 2
        self.STORAGE = 3
        self.PLAYER = 4
        self.BOX_ON_STORAGE = 5

        self.actions = {
            0: (-1, 0),  # UP
            1: (1, 0),  # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),  # RIGHT
        }

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.grid[1:-1, 1:-1] = self.FLOOR

        # Ensure there's at least one clear path
        self.grid[1:-1, 2] = self.FLOOR
        self.grid[2, 1:-1] = self.FLOOR

        # Randomly place player
        player_positions = list(zip(*np.where(self.grid == self.FLOOR)))
        self.player_pos = np.array(
            player_positions[np.random.choice(len(player_positions))]
        )
        self.grid[self.player_pos[0], self.player_pos[1]] = self.PLAYER

        # Randomly place box
        available_positions = list(zip(*np.where(self.grid == self.FLOOR)))
        self.box_pos = np.array(
            available_positions[np.random.choice(len(available_positions))]
        )
        self.grid[self.box_pos[0], self.box_pos[1]] = self.BOX

        # Randomly place storage
        available_positions = list(zip(*np.where(self.grid == self.FLOOR)))
        self.storage_pos = np.array(
            available_positions[np.random.choice(len(available_positions))]
        )
        self.grid[self.storage_pos[0], self.storage_pos[1]] = self.STORAGE

        return self.grid.copy(), {}

    def step(self, action):
        move = self.actions[action]
        new_pos = self.player_pos + move

        if self.grid[new_pos[0], new_pos[1]] == self.WALL:
            return self.grid.copy(), -1, False, False, {}

        if np.array_equal(new_pos, self.box_pos):
            new_box_pos = self.box_pos + move
            if self.grid[new_box_pos[0], new_box_pos[1]] in [self.WALL, self.BOX]:
                return self.grid.copy(), -1, False, False, {}
            self.grid[self.box_pos[0], self.box_pos[1]] = self.FLOOR
            self.box_pos = new_box_pos
            self.grid[new_box_pos[0], new_box_pos[1]] = (
                self.BOX
                if not np.array_equal(new_box_pos, self.storage_pos)
                else self.BOX_ON_STORAGE
            )

        self.grid[self.player_pos[0], self.player_pos[1]] = self.FLOOR
        self.player_pos = new_pos
        self.grid[new_pos[0], new_pos[1]] = self.PLAYER

        done = np.array_equal(self.box_pos, self.storage_pos)
        reward = 10 if done else -1

        return self.grid.copy(), reward, done, False, {}

    def render(self, mode="human"):
        if mode == "human":
            for row in self.grid:
                print(
                    " ".join(
                        [
                            "#"
                            if cell == self.WALL
                            else "$"
                            if cell == self.BOX
                            else "."
                            if cell == self.STORAGE
                            else "@"
                            if cell == self.PLAYER
                            else "*"
                            if cell == self.BOX_ON_STORAGE
                            else " "
                            for cell in row
                        ]
                    )
                )
            print("\n")
        elif mode == "rgb_array":
            cell_size = 30
            colors = {
                self.WALL: (128, 128, 128),
                self.FLOOR: (255, 255, 255),
                self.BOX: (165, 42, 42),
                self.STORAGE: (0, 255, 0),
                self.PLAYER: (0, 0, 255),
                self.BOX_ON_STORAGE: (255, 165, 0),
            }
            image = np.zeros(
                (self.height * cell_size, self.width * cell_size, 3), dtype=np.uint8
            )
            for i in range(self.height):
                for j in range(self.width):
                    color = colors[self.grid[i][j]]
                    image[
                        i * cell_size : (i + 1) * cell_size,
                        j * cell_size : (j + 1) * cell_size,
                    ] = color
            return image

    def get_state(self):
        return (tuple(self.player_pos), tuple(self.box_pos), tuple(self.storage_pos))
