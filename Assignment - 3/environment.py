import gym
from gym import spaces
import numpy as np
import pygame
import time

class MultiAgentGymEnv(gym.Env):
    """
    Multi-Agent Environment compatible with OpenAI Gym.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=(10, 10), num_agents=4, init_positions={}, goals={}, obstacles=None, cell_size=50):
        super(MultiAgentGymEnv, self).__init__()

        self.grid_rows, self.grid_cols = grid_size
        self.cell_size = cell_size
        self.num_agents = num_agents

        self.obstacles = set(obstacles) if obstacles is not None else set()
        self.agent_positions = {}
        self.goal_positions = goals
        self.init_positions = init_positions

        self.colors = {
            "background": (240, 240, 240),
            "grid": (200, 200, 200),
            "obstacle": (100, 100, 100),
            "agent": [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)],
            "goal": [(0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0)],
        }

        self.done_flags = {agent_id: False for agent_id in range(num_agents)}

        # Define action and observation spaces
        self.action_space = spaces.Dict({
            agent_id: spaces.Discrete(5) for agent_id in range(num_agents)
        })  # Actions: 0 = Stay, 1 = Up, 2 = Down, 3 = Left, 4 = Right

        self.observation_space = spaces.Dict({
            agent_id: spaces.Box(
                low=0, high=max(self.grid_rows, self.grid_cols),
                shape=(4,), dtype=np.int32
            ) for agent_id in range(num_agents)
        })

        self.reset()

        # Initialize Pygame for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((self.grid_cols * self.cell_size, self.grid_rows * self.cell_size))
        pygame.display.set_caption("Multi-Agent Environment")
        self.clock = pygame.time.Clock()

    def reset(self):
        """
        Resets the environment and returns the initial observations.
        """
        self.agent_positions, self.goal_positions = self._generate_positions()
        self.done_flags = {agent_id: False for agent_id in range(self.num_agents)}

        return self._get_observations()

    def step(self, actions):
        """
        Perform a step in the environment based on the actions.

        :param actions: Dictionary of actions for each agent.
        :return: Tuple (observations, rewards, done, info)
        """
        rewards = {agent_id: -0.1 for agent_id in range(self.num_agents)}  # Small penalty for each step
        for agent_id, action in actions.items():
            if not self.done_flags[agent_id]:
                new_position = self._compute_new_position(self.agent_positions[agent_id], action)

                if self._is_valid_position(new_position):
                    self.agent_positions[agent_id] = new_position

                if self.agent_positions[agent_id] == self.goal_positions[agent_id]:
                    self.done_flags[agent_id] = True
                    rewards[agent_id] = 10  # Reward for reaching the goal

        done = all(self.done_flags.values())
        if done:
            for agent_id in rewards:
                rewards[agent_id] += 100  # Final reward for completing the task

        return self._get_observations(), rewards, done, {}

    def render(self, mode='human'):
        """
        Render the environment using Pygame.
        """
        self.screen.fill(self.colors["background"])

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.colors["grid"], rect, 1)

                if (row, col) in self.obstacles:
                    pygame.draw.rect(self.screen, self.colors["obstacle"], rect)

        for agent_id, (r, c) in self.goal_positions.items():
            goal_rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors["goal"][agent_id % len(self.colors["goal"])], goal_rect)

        for agent_id, (r, c) in self.agent_positions.items():
            pygame.draw.circle(
                self.screen,
                self.colors["agent"][agent_id % len(self.colors["agent"])],
                (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2),
                self.cell_size // 3
            )

        pygame.display.flip()

    def close(self):
        """
        Close the rendering window.
        """
        pygame.quit()

    def _generate_positions(self):
        """
        Generate initial positions for agents and goals.
        """
        available_positions = [
            (r, c) for r in range(self.grid_rows) for c in range(self.grid_cols)
            if (r, c) not in self.obstacles
        ]
        np.random.shuffle(available_positions)

        if not self.goal_positions:
            goal_positions = {i: available_positions.pop() for i in range(self.num_agents)}
        else:
            goal_positions = self.goal_positions
        
        if not self.init_positions:
            agent_positions = {i: available_positions.pop() for i in range(self.num_agents)}
        else:
            agent_positions = self.init_positions

        return agent_positions, goal_positions

    def _get_observations(self):
        """
        Create observations for all agents.
        """
        observations = {}
        for agent_id in range(self.num_agents):
            agent_pos = self.agent_positions[agent_id]
            goal_pos = self.goal_positions[agent_id]
            observations[agent_id] = np.array([*agent_pos, *goal_pos], dtype=np.int32)
        return observations

    def _compute_new_position(self, position, action):
        """
        Compute the new position based on the current position and action.
        """
        row, col = position
        if action == 0:  # Stay
            return row, col
        elif action == 1:  # Up
            return row - 1, col
        elif action == 2:  # Down
            return row + 1, col
        elif action == 3:  # Left
            return row, col - 1
        elif action == 4:  # Right
            return row, col + 1

    def _is_valid_position(self, position):
        """
        Check if a position is valid.
        """
        row, col = position
        return (
            0 <= row < self.grid_rows and
            0 <= col < self.grid_cols and
            position not in self.obstacles and
            position not in self.agent_positions.values()
        )


# Example usage:
if __name__ == "__main__":
    # Create environment
    init_positions = {
        0: (1,1),
        1: (8,1),
        2: (1,8),
        3: (8,8)
    }

    goals = {
        0: (5,8),
        1: (1,5),
        2: (8,4),
        3: (5,1)
    }
    obstacles=[
        (0, 4),
        (1, 4),
        (2, 4),
        (2, 5),
        (4, 0),
        (4, 1),
        (4, 2),
        (5, 2),
        (4, 9),
        (4, 8),
        (4, 7),
        (5, 7),
        (9, 5),
        (8, 5),
        (7, 5),
        (7, 4)
    ]
    env = MultiAgentGymEnv(init_positions=init_positions, goals=goals, obstacles=obstacles)
    
    # Reset environment
    # obs = env.reset()
    # env.render()
    
    try:
        for episode in range(1):
            # env.current_episode = episode + 1
            obs = env.reset()
            env.render()
        # Run a few random steps
            for _ in range(10000):
                action = env.action_space.sample()  # Random action
                obs, reward, done, info = env.step(action)
                env.render()  
                time.sleep(5)              
                if done:
                    print("Done")
                    # obs = env.reset()
                    break
    finally:
        env.close()