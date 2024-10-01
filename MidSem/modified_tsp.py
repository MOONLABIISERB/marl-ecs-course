import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from numpy import typing as npt

class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment with modified reward structure."""

    def __init__(
        self,
        num_targets: int = 10,
        max_area: int = 15,
        shuffle_time: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__()
        np.random.seed(seed)

        self.steps: int = 0
        self.episodes: int = 0
        self.shuffle_time: int = shuffle_time
        self.num_targets: int = num_targets
        self.max_steps: int = num_targets
        self.max_area: int = max_area

        self.locations: npt.NDArray[np.float32] = self._generate_points(self.num_targets)
        self.distances: npt.NDArray[np.float32] = self._calculate_distances(self.locations)
        self.distances /= np.max(self.distances)  # Normalize distances

        self.initial_profits: npt.NDArray[np.float32] = (np.arange(1, self.num_targets + 1, dtype=np.float32) * 10.0)
        self.current_profits: npt.NDArray[np.float32] = self.initial_profits.copy()
        self.current_profits /= np.max(self.current_profits)  # Normalize profits

        self.observation_space = gym.spaces.Discrete(self.num_targets)
        self.action_space = gym.spaces.Discrete(self.num_targets)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[int, Dict[str, None]]:
        """Reset environment for the start of a new episode."""
        self.steps: int = 0
        self.episodes += 1

        self.loc: int = 0
        self.visited_targets: npt.NDArray[np.float32] = np.zeros(self.num_targets)
        self.current_profits = self.initial_profits.copy()

        if self.episodes % self.shuffle_time == 0:
            np.random.shuffle(self.initial_profits)

        return self.loc, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, None]]:
        """Step function for transitioning between states."""
        self.steps += 1
        past_loc = self.loc
        next_loc = action

        self.current_profits -= self.distances[past_loc, next_loc]  # Decay profits based on travel distance

        reward = self._get_rewards(next_loc)

        self.visited_targets[next_loc] = 1

        terminated = bool(self.steps == self.max_steps)
        truncated = False

        self.loc = next_loc
        return self.loc, reward, terminated, truncated, {}

    def _generate_points(self, num_points: int) -> npt.NDArray[np.float32]:
        """Generate random 2D points for target locations."""
        return np.random.uniform(low=0, high=self.max_area, size=(num_points, 2)).astype(np.float32)

    def _calculate_distances(self, locations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Calculate Euclidean distances between target locations."""
        n = len(locations)
        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, next_loc: int) -> float:
        """Reward function based on distance traveled and profit decay."""
        revisit_penalty = -100 if self.visited_targets[next_loc] else 0
        distance_penalty = -0.2 * self.distances[self.loc, next_loc]
        reward_for_visiting = self.current_profits[next_loc] if not self.visited_targets[next_loc] else 0
        step_penalty = -1

        # Calculate total reward for this step
        total_reward = reward_for_visiting + distance_penalty + revisit_penalty + step_penalty

        # Clip rewards to avoid extreme values
        return np.clip(total_reward, -500, 500)
