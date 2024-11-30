import numpy as np
import pygame
from typing import List, Tuple, Dict
from enum import Enum


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4


class MAPFEnvironment:
    # Colors for visualization
    COLORS = {
        "background": (255, 255, 255),  # White
        "grid_lines": (200, 200, 200),  # Light gray
        "obstacle": (128, 128, 128),  # Gray
        "agents": {
            0: (144, 238, 144),  # Light green
            1: (173, 216, 230),  # Light blue
            2: (255, 255, 153),  # Light yellow
            3: (216, 191, 216),  # Light purple
        },
        "goals": {
            0: (0, 255, 0),  # Green
            1: (0, 0, 255),  # Blue
            2: (255, 255, 0),  # Yellow
            3: (128, 0, 128),  # Purple
        },
    }

    def __init__(self, grid_size: int = 10, cell_size: int = 60, use_pygame: bool = True):
        """
        Initialize the MAPF environment.

        Args:
            grid_size (int): Size of the grid (default 10x10)
            cell_size (int): Size of each cell in pixels for visualization
            use_pygame (bool): Whether to initialize Pygame for visualization
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)

        # Initialize agents, goals, and obstacles
        self.agents_pos = {
            0: (1, 1),  # Green agent
            1: (8, 1),  # Blue agent
            2: (8, 8),  # Yellow agent
            3: (1, 8),  # Purple agent
        }

        self.goals = {
            0: (5, 8),  # Green goal
            1: (1, 5),  # Blue goal
            2: (5, 1),  # Yellow goal
            3: (8, 4),  # Purple goal
        }

        # Set up obstacles (1 represents obstacle)
        self.obstacles = [
            (0, 4),
            (1, 4),
            (2, 4),  # Vertical obstacle middle
            (2, 5),
            (4, 0),  # Left horizontal obstacle
            (4, 1),
            (4, 2),
            (5, 2),  # Right horizontal obstacle
            (9, 5),
            (8, 5),
            (7, 5),
            (7, 4),
            (4, 9),
            (4, 8),
            (4, 7),
            (5, 7),
        ]

        self._place_obstacles()
        self.num_agents = len(self.agents_pos)
        self.steps = 0
        self.agents_done = set()

        # Initialize Pygame only if needed
        self.use_pygame = use_pygame
        if use_pygame:
            pygame.init()
            self.window_size = grid_size * cell_size
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Multi-Agent Path Finding")
            self.font = pygame.font.Font(None, 36)

    def get_state_copy(self):
        """Create a copy of the environment without Pygame elements."""
        env_copy = MAPFEnvironment(self.grid_size, self.cell_size, use_pygame=False)
        env_copy.grid = self.grid.copy()
        env_copy.agents_pos = self.agents_pos.copy()
        env_copy.goals = self.goals.copy()
        env_copy.obstacles = self.obstacles.copy()
        env_copy.num_agents = self.num_agents
        env_copy.steps = self.steps
        env_copy.agents_done = self.agents_done.copy()
        return env_copy

    def _place_obstacles(self):
        """Place obstacles in the grid"""
        for obs in self.obstacles:
            self.grid[obs] = 1

    def _is_valid_move(self, pos: Tuple[int, int], action: Action) -> Tuple[bool, Tuple[int, int]]:
        """Check if a move is valid and return the new position."""
        x, y = pos

        if action == Action.STAY:
            return True, (x, y)

        # Calculate new position
        if action == Action.UP:
            new_pos = (x - 1, y)
        elif action == Action.DOWN:
            new_pos = (x + 1, y)
        elif action == Action.LEFT:
            new_pos = (x, y - 1)
        elif action == Action.RIGHT:
            new_pos = (x, y + 1)

        # Check bounds
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            return False, pos

        # Check obstacles
        if self.grid[new_pos] == 1:
            return False, pos

        return True, new_pos

    def _check_collisions(self, current_positions: Dict[int, Tuple[int, int]], new_positions: Dict[int, Tuple[int, int]]) -> bool:
        """
        Check for both destination conflicts and path crossing conflicts.

        Args:
            current_positions: Dictionary of current agent positions
            new_positions: Dictionary of proposed new positions

        Returns:
            bool: True if there's a collision, False otherwise
        """
        # Check destination conflicts (two agents trying to occupy the same cell)
        position_count = {}
        for agent_id, pos in new_positions.items():
            if pos in position_count:
                return True
            position_count[pos] = agent_id

        # Check path crossing conflicts (agents trying to swap positions)
        for agent1 in current_positions:
            for agent2 in current_positions:
                if agent1 < agent2:  # Check each pair only once
                    if current_positions[agent1] == new_positions[agent2] and current_positions[agent2] == new_positions[agent1]:
                        return True

        return False

    def step(self, actions: Dict[int, Action]) -> Tuple[Dict, float, bool, dict]:
        """Take a step in the environment."""
        self.steps += 1
        reward = -1  # -1 reward per step
        collision_penalty = -5  # Additional penalty for attempting invalid moves

        # Store current positions
        current_positions = self.agents_pos.copy()
        new_positions = {}

        # First, check all moves are valid and get new positions
        for agent_id, action in actions.items():
            if agent_id in self.agents_done:
                new_positions[agent_id] = current_positions[agent_id]
                continue

            is_valid, new_pos = self._is_valid_move(current_positions[agent_id], action)
            new_positions[agent_id] = new_pos

        # Check for any type of collision
        if self._check_collisions(current_positions, new_positions):
            # If there's a collision, no agents move and we apply a penalty
            new_positions = current_positions
            reward += collision_penalty

        # Update positions
        self.agents_pos = new_positions

        # Check which agents reached their goals
        for agent_id, pos in self.agents_pos.items():
            if pos == self.goals[agent_id]:
                self.agents_done.add(agent_id)

        # Check if all agents reached their goals
        done = len(self.agents_done) == self.num_agents

        info = {"steps": self.steps, "agents_done": self.agents_done}

        return self.agents_pos, reward, done, info

    def reset(self) -> Dict[int, Tuple[int, int]]:
        """Reset the environment to initial state."""
        self.steps = 0
        self.agents_done = set()

        # Reset agent positions to initial state
        self.agents_pos = {
            0: (2, 1),  # Green agent
            1: (7, 1),  # Blue agent
            2: (7, 8),  # Yellow agent
            3: (2, 8),  # Purple agent
        }

        return self.agents_pos

    def get_valid_actions(self, agent_id: int) -> List[Action]:
        """Get list of valid actions for an agent."""
        valid_actions = []
        current_pos = self.agents_pos[agent_id]

        for action in Action:
            is_valid, _ = self._is_valid_move(current_pos, action)
            if is_valid:
                valid_actions.append(action)

        return valid_actions

    def render(self, mode="human"):
        """Render the environment using Pygame."""
        # Fill background
        self.screen.fill(self.COLORS["background"])

        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical lines
            pygame.draw.line(self.screen, self.COLORS["grid_lines"], (i * self.cell_size, 0), (i * self.cell_size, self.window_size))
            # Horizontal lines
            pygame.draw.line(self.screen, self.COLORS["grid_lines"], (0, i * self.cell_size), (self.window_size, i * self.cell_size))

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(
                self.screen, self.COLORS["obstacle"], (obs[1] * self.cell_size, obs[0] * self.cell_size, self.cell_size, self.cell_size)
            )

        # Draw goals
        for agent_id, goal_pos in self.goals.items():
            # Draw plus sign
            center_x = goal_pos[1] * self.cell_size + self.cell_size // 2
            center_y = goal_pos[0] * self.cell_size + self.cell_size // 2
            size = self.cell_size // 4

            pygame.draw.line(self.screen, self.COLORS["goals"][agent_id], (center_x - size, center_y), (center_x + size, center_y), 3)
            pygame.draw.line(self.screen, self.COLORS["goals"][agent_id], (center_x, center_y - size), (center_x, center_y + size), 3)

        # Draw agents
        for agent_id, pos in self.agents_pos.items():
            pygame.draw.rect(
                self.screen,
                self.COLORS["agents"][agent_id],
                (pos[1] * self.cell_size + 2, pos[0] * self.cell_size + 2, self.cell_size - 4, self.cell_size - 4),
            )

        # print the step on the screen
        text = self.font.render(f"Step: {self.steps}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        # Update display
        pygame.display.flip()

    def close(self):
        """Close the Pygame window."""
        pygame.quit()


def main():
    """Test the environment with random actions."""
    env = MAPFEnvironment()
    env.reset()

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Take random actions for all agents
                    actions = {agent_id: np.random.choice(list(Action)) for agent_id in range(env.num_agents)}
                    next_states, reward, done, info = env.step(actions)
                    print(f"Step {info['steps']}, Reward: {reward}, Done: {done}")
                    if done:
                        env.reset()

        env.render()
        clock.tick(10)

    env.close()


if __name__ == "__main__":
    main()
