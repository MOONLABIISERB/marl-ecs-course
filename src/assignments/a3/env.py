import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    Multi-agent grid world environment where agents need to reach their goal positions
    while avoiding collisions with other agents and walls.
    """

    def __init__(
        self,
        grid_size=8,
        n_agents=3,
        walls=None,
        goal_positions=None,
        random_start=True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.random_start = random_start

        # Initialize walls
        if walls is None:
            self.walls = []
        else:
            self.walls = walls

        # Initialize goal positions
        if goal_positions is None:
            self.goal_positions = np.array(
                [[self.grid_size - 1, self.grid_size - 1] for _ in range(n_agents)]
            )
        else:
            assert (
                len(goal_positions) == n_agents
            ), "Must provide goal positions for all agents"
            self.goal_positions = np.array(goal_positions)

        # Validate goal positions are not in walls
        for goal in self.goal_positions:
            assert not any(
                np.array_equal(goal, wall) for wall in self.walls
            ), "Goal position cannot be in a wall"

        # Action space: 0: up, 1: right, 2: down, 3: left, 4: stay
        self.action_space = spaces.Discrete(5)

        # Observation space: position of all agents + goals + walls
        self.observation_space = spaces.Box(
            low=0,
            high=grid_size - 1,
            shape=(
                n_agents * 4 + len(self.walls) * 2,
            ),  # (x,y) for agents + goals + walls
            dtype=np.float32,
        )

        # Initialize positions
        self.reset()

    def set_goal_positions(self, goal_positions):
        """Set new goal positions for all agents."""
        assert (
            len(goal_positions) == self.n_agents
        ), "Must provide goal positions for all agents"
        self.goal_positions = np.array(goal_positions)
        # Validate goal positions are not in walls
        for goal in self.goal_positions:
            assert not any(
                np.array_equal(goal, wall) for wall in self.walls
            ), "Goal position cannot be in a wall"

    def set_start_positions(self, start_positions):
        """Set specific start positions for all agents."""
        assert (
            len(start_positions) == self.n_agents
        ), "Must provide start positions for all agents"
        self.agent_positions = np.array(start_positions)
        # Validate start positions are valid
        self._validate_positions(self.agent_positions)

    def _validate_positions(self, positions):
        """Check if positions are valid (not in walls and within grid)."""
        for pos in positions:
            # Check if position is within grid bounds
            assert (
                0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
            ), f"Position {pos} is outside grid bounds"
            # Check if position overlaps with wall
            assert not any(
                np.array_equal(pos, wall) for wall in self.walls
            ), f"Position {pos} overlaps with a wall"

    def reset(self, seed=None):
        super().reset(seed=seed)

        if self.random_start:
            # Randomly place agents
            positions = []
            self.agent_positions = np.zeros((self.n_agents, 2))

            for i in range(self.n_agents):
                attempts = 0
                max_attempts = 100
                while attempts < max_attempts:
                    pos = self.np_random.integers(0, self.grid_size, size=2)
                    # Check if position is valid (not in walls or other agents)
                    if not any(
                        np.array_equal(pos, wall) for wall in self.walls
                    ) and not any(np.array_equal(pos, p) for p in positions):
                        self.agent_positions[i] = pos
                        positions.append(pos)
                        break
                    attempts += 1
                if attempts == max_attempts:
                    raise RuntimeError("Could not find valid positions for all agents")
        else:
            # Start at default positions if not set
            self.agent_positions = np.array([[0, 0] for _ in range(self.n_agents)])

        return self._get_obs(), {}

    def _get_obs(self):
        # Concatenate agent positions, goal positions, and wall positions
        obs = np.concatenate(
            [
                self.agent_positions.flatten(),
                self.goal_positions.flatten(),
                np.array(self.walls).flatten() if self.walls else np.array([]),
            ]
        )
        return obs.astype(np.float32)

    def is_valid_move(self, position):
        """Check if a position is valid (within bounds and not in a wall)."""
        x, y = position
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        return not any(np.array_equal(position, wall) for wall in self.walls)

    def step(self, actions):
        # Save old positions
        old_positions = self.agent_positions.copy()

        # Update positions based on actions
        for i, action in enumerate(actions):
            new_position = self.agent_positions[i].copy()

            if action == 0:  # up
                new_position[1] += 1
            elif action == 1:  # right
                new_position[0] += 1
            elif action == 2:  # down
                new_position[1] -= 1
            elif action == 3:  # left
                new_position[0] -= 1
            # action == 4 is stay, no position update needed

            # Only update if the move is valid
            if self.is_valid_move(new_position):
                self.agent_positions[i] = new_position

        # Check for collisions between agents
        collisions = False
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.array_equal(self.agent_positions[i], self.agent_positions[j]):
                    collisions = True
                    self.agent_positions = old_positions  # Revert moves
                    break
            if collisions:
                break

        # Calculate rewards and check if done
        rewards = np.zeros(self.n_agents)
        at_goal = np.zeros(self.n_agents, dtype=bool)

        for i in range(self.n_agents):
            # Distance-based reward
            curr_dist = np.linalg.norm(self.agent_positions[i] - self.goal_positions[i])
            old_dist = np.linalg.norm(old_positions[i] - self.goal_positions[i])
            rewards[i] = old_dist - curr_dist

            # Check if agent is at goal
            if np.array_equal(self.agent_positions[i], self.goal_positions[i]):
                at_goal[i] = True
                rewards[i] += 0.5  # Small continuous reward for staying at goal

            # Collision penalty
            if collisions:
                rewards[i] -= 1000

            # Wall collision penalty (if attempted to move into wall)
            if not np.array_equal(old_positions[i], self.agent_positions[i]):
                rewards[i] -= 1  # Small penalty for movement to encourage efficiency

        # Success only when all agents reach their goals
        all_at_goal = np.all(at_goal)
        if all_at_goal:
            # Give large reward to all agents for collective success
            rewards += 500

        # Episode done if all agents reach goals or there's a collision
        done = all_at_goal
        terminated = collisions

        return self._get_obs(), rewards, done, terminated, {}


# Example usage:
if __name__ == "__main__":
    # Define walls (list of [x, y] coordinates)
    walls = [
        [2, 2],
        [2, 3],
        [2, 4],  # Vertical wall
        [4, 4],
        [5, 4],
        [6, 4],  # Horizontal wall
    ]

    # Define goal positions for each agent
    goal_positions = [
        [7, 7],  # Goal for agent 1
        [7, 6],  # Goal for agent 2
        [6, 7],  # Goal for agent 3
    ]

    # Create environment with walls and custom goal positions
    env = GridWorldEnv(
        grid_size=8,
        n_agents=3,
        walls=walls,
        goal_positions=goal_positions,
        random_start=True,  # Set to False for fixed starting positions
    )

    # If you want to set specific starting positions
    start_positions = [
        [0, 0],  # Start position for agent 1
        [0, 1],  # Start position for agent 2
        [1, 0],  # Start position for agent 3
    ]
    env.set_start_positions(start_positions)

    # Reset environment and get initial observation
    obs, _ = env.reset()
    print("Initial observation:", obs)
