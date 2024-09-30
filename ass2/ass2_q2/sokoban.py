import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CustomSokobanEnv(gym.Env):
    """Custom Sokoban environment where a player pushes boxes to storage locations."""

    def __init__(self):
        super(CustomSokobanEnv, self).__init__()

        # Dimensions of the grid
        self.height = 6
        self.width = 7

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(self.height, self.width), dtype=np.uint8
        )

        # Define object types
        self.WALL = 0
        self.FLOOR = 1
        self.BOX = 2
        self.STORAGE = 3
        self.PLAYER = 4
        self.BOX_ON_STORAGE = 5

        # Map actions to movement (row, col)
        self.moves = {
            0: (-1, 0),  # UP
            1: (1, 0),  # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),  # RIGHT
        }

        self.reset()

    def reset(self):
        """Reset the environment to its initial state."""
        # Create empty grid
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.grid[1:-1, 1:-1] = self.FLOOR

        # Make sure there's a clear path
        self.grid[1:-1, 2] = self.FLOOR
        self.grid[2, 1:-1] = self.FLOOR

        # Place player at a random floor position
        floor_positions = list(zip(*np.where(self.grid == self.FLOOR)))
        self.player_pos = np.array(floor_positions[np.random.choice(len(floor_positions))])
        self.grid[self.player_pos[0], self.player_pos[1]] = self.PLAYER

        # Place the box at a random floor position
        available_positions = list(zip(*np.where(self.grid == self.FLOOR)))
        self.box_pos = np.array(available_positions[np.random.choice(len(available_positions))])
        self.grid[self.box_pos[0], self.box_pos[1]] = self.BOX

        # Place the storage at a random floor position
        available_positions = list(zip(*np.where(self.grid == self.FLOOR)))
        self.storage_pos = np.array(available_positions[np.random.choice(len(available_positions))])
        self.grid[self.storage_pos[0], self.storage_pos[1]] = self.STORAGE

        return self.grid.copy(), {}

    def step(self, action):
        """Take a step in the environment given the player's action."""
        movement = self.moves[action]
        next_pos = self.player_pos + movement

        # Check if next position is a wall
        if self.grid[next_pos[0], next_pos[1]] == self.WALL:
            return self.grid.copy(), -1, False, False, {}

        # If the next position is the box, try to push the box
        if np.array_equal(next_pos, self.box_pos):
            new_box_pos = self.box_pos + movement
            if self.grid[new_box_pos[0], new_box_pos[1]] in [self.WALL, self.BOX]:
                return self.grid.copy(), -1, False, False, {}
            self.grid[self.box_pos[0], self.box_pos[1]] = self.FLOOR
            self.box_pos = new_box_pos
            self.grid[new_box_pos[0], new_box_pos[1]] = (
                self.BOX if not np.array_equal(new_box_pos, self.storage_pos) else self.BOX_ON_STORAGE
            )

        # Update player position
        self.grid[self.player_pos[0], self.player_pos[1]] = self.FLOOR
        self.player_pos = next_pos
        self.grid[next_pos[0], next_pos[1]] = self.PLAYER

        # Check if the box is on the storage location
        is_done = np.array_equal(self.box_pos, self.storage_pos)
        reward = 10 if is_done else -1

        return self.grid.copy(), reward, is_done, False, {}

    def render(self, mode="human"):
        """Render the environment in text or image mode."""
        if mode == "human":
            for row in self.grid:
                print(
                    " ".join(
                        [
                            "#" if cell == self.WALL
                            else "$" if cell == self.BOX
                            else "." if cell == self.STORAGE
                            else "@" if cell == self.PLAYER
                            else "*" if cell == self.BOX_ON_STORAGE
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
            image = np.zeros((self.height * cell_size, self.width * cell_size, 3), dtype=np.uint8)
            for i in range(self.height):
                for j in range(self.width):
                    color = colors[self.grid[i][j]]
                    image[
                        i * cell_size : (i + 1) * cell_size,
                        j * cell_size : (j + 1) * cell_size,
                    ] = color
            return image

    def get_state(self):
        """Return the current state as a tuple."""
        return (tuple(self.player_pos), tuple(self.box_pos), tuple(self.storage_pos))
