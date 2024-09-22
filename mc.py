
"""Environment for Travelling Salesman Problem."""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from collections import defaultdict



class TSP(gym.Env):
    """Traveling Salesman Problem (TSP) RL environment for persistent monitoring.

    The agent navigates a set of targets based on precomputed distances. It aims to visit
    all targets in the least number of steps, with rewards determined by the distance traveled.
    """

    def __init__(self, num_targets: int, max_area: int = 30, seed: int = None) -> None:
        """Initialize the TSP environment.

        Args:
            num_targets (int): Number of targets the agent needs to visit.
            max_area (int): Max Square area where the targets are defined. Defaults to 30
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super().__init__()
        if seed is not None:
            np.random.seed(seed=seed)

        self.steps: int = 0
        self.num_targets: int = num_targets

        self.max_steps: int = num_targets
        self.max_area: int = max_area

        self.locations: np.ndarray = self._generate_points(self.num_targets)
        self.distances: np.ndarray = self._calculate_distances(self.locations)

        # Observation Space : {current loc (loc), dist_array (distances), coordintates (locations)}
        self.obs_low = np.concatenate(
            [
                np.array([0], dtype=np.float32),
                np.zeros(self.num_targets, dtype=np.float32),
                np.zeros(2 * self.num_targets, dtype=np.float32),
            ]
        )

        self.obs_high = np.concatenate(
            [
                np.array([self.num_targets], dtype=np.float32),
                2 * self.max_area * np.ones(self.num_targets, dtype=np.float32),
                self.max_area * np.ones(2 * self.num_targets, dtype=np.float32),
            ]
        )

        # Action Space : {next_target}
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Discrete(self.num_targets)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, None]]:
        """Reset the environment to the initial state.

        Args:
            seed (Optional[int], optional): Seed to reset the environment. Defaults to None.
            options (Optional[dict], optional): Additional reset options. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict[str, None]]: The initial state of the environment and an empty info dictionary.
        """
        self.steps: int = 0

        self.loc: int = 0
        self.visited_targets: List = []
        self.dist: List = self.distances[self.loc]

        state = np.concatenate(
            (
                np.array([self.loc]),
                np.array(self.dist),
                np.array(self.locations).reshape(-1),
            ),
            dtype=np.float32,
        )
        return state, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
        """Take an action (move to the next target).

        Args:
            action (int): The index of the next target to move to.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
                - The new state of the environment.
                - The reward for the action.
                - A boolean indicating whether the episode has terminated.
                - A boolean indicating if the episode is truncated.
                - An empty info dictionary.
        """
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        reward = self._get_rewards(past_loc, next_loc)
        self.visited_targets.append(next_loc)

        next_dist = self.distances[next_loc]
        terminated = bool(self.steps == self.max_steps)
        truncated = False

        next_state = np.concatenate(
            [
                np.array([next_loc]),
                next_dist,
                np.array(self.locations).reshape(-1),
            ],
            dtype=np.float32,
        )

        self.loc, self.dist = next_loc, next_dist
        return (next_state, reward, terminated, truncated, {})

    def _generate_points(self, num_points: int) -> np.ndarray:
        """Generate random 2D points representing target locations.

        Args:
            num_points (int): Number of points to generate.

        Returns:
            np.ndarray: Array of 2D coordinates for each target.
        """
        points = []
        # Generate n random 2D points within the 10x10 grid
        while len(points) < num_points:
            x = np.random.random() * self.max_area
            y = np.random.random() * self.max_area
            if [x, y] not in points:
                points.append([x, y])

        return np.array(points)

    def _calculate_distances(self, locations: List) -> float:
        """Calculate the distance matrix between all target locations.

        Args:
            locations (List): List of 2D target locations.

        Returns:
            np.ndarray: Matrix of pairwise distances between targets.
        """
        n = len(locations)

        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, past_loc: int, next_loc: int) -> float:
        """Calculate the reward based on the distance traveled, however if a target gets visited again then it incurs a high penalty.

        Args:
            past_loc (int): Previous location of the agent.
            next_loc (int): Next location of the agent.

        Returns:
            float: Reward based on the travel distance between past and next locations, or negative reward if repeats visit.
        """
        if next_loc not in self.visited_targets:
            reward = -self.distances[past_loc][next_loc]
        else:
            reward = -10000
        return reward


if __name__ == "__main__":
    num_targets = 50
    env = TSP(num_targets)
    num_episodes = 1000
    discount_factor = 0.9
    epsilon = 0.1  # Epsilon-greedy exploration

    # Initialize Q-values and returns for both methods
    Q_first_visit = np.zeros((env.num_targets, env.num_targets))
    Q_every_visit = np.zeros((env.num_targets, env.num_targets))
    returns_first_visit = defaultdict(list)  # To store returns for first-visit
    returns_every_visit = defaultdict(list)  # To store returns for every-visit

    # Monte Carlo with exploring starts for both first-visit and every-visit
    for method in ["first_visit", "every_visit"]:
        for episode in range(num_episodes):
            episode_history = []  # To store (state, action, reward) for the episode
            state, _ = env.reset()
            state = int(state[0])
            done = False
            total_reward = 0
            
            while not done:
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = env.action_space.sample()  # Explore
                else:
                    action = np.argmax(Q_first_visit[state]) if method == "first_visit" else np.argmax(Q_every_visit[state])
                
                # Take action and observe reward and next state
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = int(next_state[0])
                episode_history.append((state, action, reward))  # Store state-action-reward
                state = next_state
                total_reward += reward
                done = terminated or truncated

            # Calculate the returns and update Q-values
            G = 0  # Return
            
            # Loop through the episode history in reverse
            for state, action, reward in reversed(episode_history):
                G = reward + discount_factor * G
                
                if method == "first_visit":
                    # First-visit MC
                    if (state, action) not in returns_first_visit:  # Check if first visit
                        returns_first_visit[(state, action)].append(G)
                        Q_first_visit[state, action] = np.mean(returns_first_visit[(state, action)])  # Average return

                elif method == "every_visit":
                    # Every-visit MC
                    returns_every_visit[(state, action)].append(G)
                    Q_every_visit[state, action] = np.mean(returns_every_visit[(state, action)])  # Average return

            if episode % 100 == 0:
                print(f"[{method}] Episode {episode}, Total reward: {total_reward}")

    # Extract the optimal policy after convergence
    policy_first_visit = np.argmax(Q_first_visit, axis=1)
    policy_every_visit = np.argmax(Q_every_visit, axis=1)

    print("Optimal Policy (First Visit):", policy_first_visit)
    print("Optimal Policy (Every Visit):", policy_every_visit)