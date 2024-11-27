import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import pygame
import sys

from agent import TetheredBoatsAgent


class TetheredBoatsEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        grid_size=10,
        n_boats=2,
        tether_length=3,
        time_penalty=-1,
        trash_reward=10,
        complete_reward=50,
        incomplete_penalty=-50,
        invalid_move_penalty=-10,
        step_per_episode=100,
        num_episode=1,
        seed=None,
        n_trash=10,
    ):
        super(TetheredBoatsEnv, self).__init__()

        # Pygame specific settings
        self.WINDOW_SIZE = 800  # pixels
        self.HUD_HEIGHT = 100  # pixels for HUD
        self.CELL_SIZE = self.WINDOW_SIZE // grid_size
        self.screen = None
        self.clock = None
        self.font = None

        if seed is not None:
            np.random.seed(seed)

        # RL parameters
        self.step_per_episode = step_per_episode
        self.num_episode = num_episode

        # Track cumulative reward
        self.cumulative_reward = 0
        self.current_episode = 1

        # Environment parameters
        self.grid_size = grid_size
        self.n_boats = n_boats
        self.tether_length = tether_length
        self.n_trash = n_trash

        # Rewards
        self.time_penalty = time_penalty
        self.trash_reward = trash_reward
        self.complete_reward = complete_reward
        self.incomplete_penalty = incomplete_penalty
        self.invalid_move_penalty = invalid_move_penalty

        # Grid values
        self.EMPTY = 0
        self.TRASH = 1
        self.BOAT_OR_TETHER = 2

        # Action space: 0 - straight, 1 - 45° left, 2 - 45° right
        # Each boat can take one of these actions
        self.action_space = spaces.MultiDiscrete([9] * n_boats)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(grid_size, grid_size), dtype=np.float32
        )

        # Initialize state
        self.reset()

    def _grid_to_pixel(self, grid_pos):
        """Convert grid coordinates to pixel coordinates"""
        x, y = grid_pos
        return (
            y * self.CELL_SIZE + self.CELL_SIZE // 2,
            x * self.CELL_SIZE + self.CELL_SIZE // 2,
        )

    def _get_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        x1, y1 = pos1
        x2, y2 = pos2

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _get_tether_cells(self, boat1_pos, boat2_pos):
        """Calculate cells occupied by tether between two boats"""
        x1, y1 = boat1_pos
        x2, y2 = boat2_pos

        # Calculate distance between boats
        distance = self._get_distance(boat1_pos, boat2_pos)

        # If distance allows for 3 points (i.e., there's room for a middle point)
        if distance < self.tether_length:  # Minimum distance needed for 3 points
            n_points = self.tether_length
        else:
            n_points = self.tether_length + 1

        # Get all points on line between boats
        points = []
        # n_points = self.tether_length + 1
        for i in range(n_points):
            t = i / (n_points - 1)
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            points.append((x, y))

        return points

    def _is_valid_move(self, current_pos, new_pos, other_boat_pos):
        """Check if move is valid (within grid, one cell distance, and tether length)"""
        x, y = new_pos

        # Check grid boundaries
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False

        # Check movement distance (can only move one cell in any direction)
        curr_x, curr_y = current_pos
        if abs(x - curr_x) > 1 or abs(y - curr_y) > 1:
            return False

        # Check tether length constraint
        new_distance = self._get_distance(new_pos, other_boat_pos)
        if new_distance > self.tether_length:
            return False

        # Check if two boats are in same location
        if new_pos == other_boat_pos:
            return False

        return True

    def _get_new_position(self, pos, action):
        """Get new position based on action"""
        x, y = pos

        # Straight movement
        if action == 0:
            return (x + 1, y)
        # 45° left
        elif action == 1:
            return (x + 1, y - 1)
        # 90 left
        elif action == 2:
            return (x, y - 1)
        # 135 degree left
        elif action == 3:
            return (x - 1, y - 1)
        # 180 degree
        elif action == 4:
            return (x - 1, y)
        # 135 degree right:
        elif action == 5:
            return (x - 1, y + 1)
        # 90 degree right:
        elif action == 6:
            return (x, y + 1)
        # 45° right
        elif action == 7:
            return (x + 1, y + 1)
        # stay
        else:
            return (x, y)

    def step(self, action):
        """Execute one time step within the environment"""
        action = [int(a) for a in action]
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.step_num += 1

        reward = self.time_penalty
        done = False

        # Store previous positions
        old_boat_positions = self.boat_positions.copy()

        # Move boats based on actions
        for i, boat_action in enumerate(action):
            new_pos = self._get_new_position(self.boat_positions[i], boat_action)
            other_boat_pos = self.boat_positions[
                (i + 1) % 2
            ]  # Position of the other boat

            # Check if move is valid including tether constraint
            if self._is_valid_move(self.boat_positions[i], new_pos, other_boat_pos):
                self.boat_positions[i] = new_pos
            else:
                reward += self.invalid_move_penalty

        # Calculate tether positions
        tether_cells = self._get_tether_cells(
            self.boat_positions[0], self.boat_positions[1]
        )

        # Update grid
        self.grid.fill(self.EMPTY)

        # Place remaining trash
        for trash_pos in self.trash_positions:
            self.grid[trash_pos] = self.TRASH

        # Check for trash collection (boat or tether touching trash)
        trash_collected = []
        for trash_pos in self.trash_positions:
            tx, ty = trash_pos

            # Check if boats collect trash
            for boat_pos in self.boat_positions:
                bx, by = boat_pos
                if (tx, ty) == (bx, by):
                    trash_collected.append(trash_pos)
                    reward += self.trash_reward

            # Check if tether collects trash
            for tether_pos in tether_cells:
                if (tx, ty) == tether_pos:
                    trash_collected.append(trash_pos)
                    reward += self.trash_reward

        # Remove collected trash
        self.trash_positions = [
            pos for pos in self.trash_positions if pos not in trash_collected
        ]

        # Place boats and tether
        for boat_pos in self.boat_positions:
            self.grid[boat_pos] = self.BOAT_OR_TETHER
        for tether_pos in tether_cells:
            self.grid[tether_pos] = self.BOAT_OR_TETHER

        # Check if all trash is collected
        if len(self.trash_positions) == 0:
            reward += self.complete_reward
            done = True

        # Give penalty if task is not complete at the end of the episode
        if self.step_num == self.step_per_episode and len(self.trash_positions) != 0:
            reward += self.incomplete_penalty

        # Update cumulative reward
        self.cumulative_reward += reward

        return self.grid.astype(np.float32) / 2.0, reward, done, {}

    def reset(self):
        """Reset the state of the environment"""

        # self.fig, self.ax = plt.subplots()
        # self.fig.set_size_inches(8, 8)

        # self.ax.clear()

        self.step_num = 0
        self.cumulative_reward = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Initialize boat positions (start from left side)
        self.boat_positions = [(0, 0), (0, self.tether_length)]

        # Initialize random trash positions (on right half of grid)
        n_trash = self.n_trash  # Number of trash pieces
        self.trash_positions = []

        # set random seed
        # np.random.seed(seed)

        while len(self.trash_positions) < n_trash:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            pos = (x, y)
            if pos not in self.trash_positions:
                self.trash_positions.append(pos)
                self.grid[pos] = self.TRASH

        # Place boats and tether
        for boat_pos in self.boat_positions:
            self.grid[boat_pos] = self.BOAT_OR_TETHER
        for tether_pos in self._get_tether_cells(
            self.boat_positions[0], self.boat_positions[1]
        ):
            self.grid[tether_pos] = self.BOAT_OR_TETHER

        # In reset() and step() methods, after returning self.grid:
        return self.grid.astype(np.float32) / 2.0  # Normalize to [0, 1]

    def _render_hud(self):
        """Render the HUD with game statistics"""
        # HUD background
        pygame.draw.rect(
            self.screen,
            (240, 240, 240),
            (0, self.WINDOW_SIZE, self.WINDOW_SIZE, self.HUD_HEIGHT),
        )
        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (0, self.WINDOW_SIZE),
            (self.WINDOW_SIZE, self.WINDOW_SIZE),
            2,
        )

        # Prepare text elements
        texts = [
            f"Episode: {self.current_episode}/{self.num_episode}",
            f"Step: {self.step_num}/{self.step_per_episode}",
            f"Reward: {self.cumulative_reward:.1f}",
            f"Trash Remaining: {len(self.trash_positions)}",
        ]

        # Render text elements
        x_positions = [20, 200, 380, 560]  # Horizontal positions for each text element
        for text, x_pos in zip(texts, x_positions):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (x_pos, 30))

    def render(self, mode="human"):
        """Render the environment using Pygame"""
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Tethered Boats Environment")
            self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)  # Initialize font

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Fill background
        self.screen.fill((255, 255, 255))  # White background

        # Draw grid lines
        for i in range(self.grid_size + 1):
            pos = i * self.CELL_SIZE
            pygame.draw.line(
                self.screen, (200, 200, 200), (pos, 0), (pos, self.WINDOW_SIZE)
            )
            pygame.draw.line(
                self.screen, (200, 200, 200), (0, pos), (self.WINDOW_SIZE, pos)
            )

        # Draw trash (red circles)
        for pos in self.trash_positions:
            pixel_pos = self._grid_to_pixel(pos)
            pygame.draw.circle(self.screen, (255, 0, 0), pixel_pos, self.CELL_SIZE // 4)

        # Draw tether (green line with points)
        if len(self.boat_positions) >= 2:
            # Draw tether line and points
            tether_positions = self._get_tether_cells(
                self.boat_positions[0], self.boat_positions[1]
            )

            # Convert all positions to pixel coordinates
            pixel_positions = [self._grid_to_pixel(pos) for pos in tether_positions]

            # Draw tether lines
            if len(pixel_positions) > 1:
                pygame.draw.lines(self.screen, (0, 255, 0), False, pixel_positions, 3)

            # Draw tether points
            for pos in pixel_positions:
                pygame.draw.circle(self.screen, (0, 200, 0), pos, self.CELL_SIZE // 6)

        # Draw boats (blue triangles)
        for boat_pos in self.boat_positions:
            pixel_pos = self._grid_to_pixel(boat_pos)

            # Calculate triangle points
            size = self.CELL_SIZE // 3
            points = [
                (pixel_pos[0], pixel_pos[1] - size),  # Top
                (pixel_pos[0] - size, pixel_pos[1] + size),  # Bottom left
                (pixel_pos[0] + size, pixel_pos[1] + size),  # Bottom right
            ]
            pygame.draw.polygon(self.screen, (0, 0, 255), points)

        self._render_hud()

        # Update display
        pygame.display.flip()
        self.clock.tick(10)  # 2 FPS to match original 0.5s pause

    def close(self):
        if self.screen is not None:
            pygame.quit()


# Example usage:
if __name__ == "__main__":
    # Create environment
    env = TetheredBoatsEnv()
    agent = TetheredBoatsAgent(env)

    # load model
    agent.load_model("tethered_boats_model.pth")

    RANDOM = True

    # Reset environment
    # obs = env.reset()
    # env.render()

    try:
        for episode in range(env.num_episode):
            env.current_episode = episode + 1
            obs = env.reset()
            env.render()
            # Run a few random steps
            for _ in range(env.step_per_episode):
                print(obs)

                if RANDOM:
                    action = env.action_space.sample()  # Random action
                else:
                    action = agent.get_action(obs, training=False)
                print(action)
                obs, reward, done, info = env.step(action)
                env.render()

                if done:
                    # obs = env.reset()
                    break
    finally:
        env.close()
