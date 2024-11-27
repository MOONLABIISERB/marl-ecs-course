from typing import Dict, List, Optional, Tuple


import gymnasium as gym
import numpy as np
from numpy import typing as npt


class ModTSP(gym.Env):
    """Travelling Salesman Problem (TSP) RL environment for maximizing profits.

    The agent navigates a set of targets based on precomputed distances. It aims to visit
    all targets so maximize profits. The profits decay with time.
    """

    def __init__(
        self,
        num_targets: int = 10,
        max_area: int = 15,
        shuffle_time: int = 1,
        seed: int = None,
    ) -> None:
        """Initialize the TSP environment.

        Args:
            num_targets (int): No. of targets the agent needs to visit.
            max_area (int): Max. Square area where the targets are defined.
            shuffle_time (int): No. of episodes after which the profits ar to be shuffled.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()

        # np.random.seed(seed)

        self.total_distance = 0

        self.steps: int = 0
        self.episodes: int = 0

        self.shuffle_time: int = shuffle_time
        self.num_targets: int = num_targets

        self.max_steps: int = num_targets
        self.max_area: int = max_area

        self.locations: npt.NDArray[np.float32] = self._generate_points(self.num_targets)
        self.distances: npt.NDArray[np.float32] = self._calculate_distances(self.locations)

        # Initialize profits for each target
        self.initial_profits: npt.NDArray[np.float32] = np.arange(1, self.num_targets + 1, dtype=np.float32) * 10.0
        self.current_profits: npt.NDArray[np.float32] = self.initial_profits.copy()

        # Observation Space : {current loc (loc), current profits, dist_array (distances), coordintates (locations), visited(one hot)}
        self.obs_low = np.concatenate(
            [
                np.array([0], dtype=np.float32),  # Current location
                np.zeros(self.num_targets, dtype=np.float32),  # Array of all current profits values
                np.zeros(self.num_targets, dtype=np.float32),  # Distance to each target from current location
                np.zeros(2 * self.num_targets, dtype=np.float32),  # Cooridinates of all targets
                np.zeros(self.num_targets, dtype=np.float32),  # Visited targets (0 for unvisited, 1 for visited)
            ]
        )

        self.obs_high = np.concatenate(
            [
                np.array([self.num_targets], dtype=np.float32),  # Current location
                100 * np.ones(self.num_targets, dtype=np.float32),  # Array of all current profits values
                2 * self.max_area * np.ones(self.num_targets, dtype=np.float32),  # Distance to each target from current location
                self.max_area * np.ones(2 * self.num_targets, dtype=np.float32),  # Cooridinates of all targets
                np.ones(self.num_targets, dtype=np.float32),  # Visited targets (0 for unvisited, 1 for visited)
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
        self.total_distance = 0
        self.steps: int = 0
        self.episodes += 1

        self.loc: int = 0
        self.visited_targets = np.zeros(self.num_targets, dtype=np.float32)
        # self.visited_targets[self.loc] = 1
        if self.episodes % self.shuffle_time == 0:
            # print("Shuffling profits")
            np.random.shuffle(self.initial_profits)
        self.current_profits = self.initial_profits.copy()

        self.dist: List = self.distances[self.loc]

        state = np.concatenate(
            (np.array([self.loc]), self.initial_profits, np.array(self.dist), np.array(self.locations).reshape(-1), self.visited_targets),
            dtype=np.float32,
        )
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, None]]:
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

        distance_travelled = self.distances[past_loc, next_loc]
        self.total_distance += distance_travelled

        self.current_profits -= self.distances[past_loc, next_loc]
        reward = self._get_rewards(next_loc)
        self.visited_targets[next_loc] = 1

        next_dist = self.distances[next_loc]
        terminated = bool(self.steps == self.max_steps)
        truncated = False

        next_state = np.concatenate(
            [
                np.array([next_loc]),
                self.current_profits,
                next_dist,
                np.array(self.locations).reshape(-1),
                self.visited_targets,
            ],
            dtype=np.float32,
        )

        self.loc, self.dist = next_loc, next_dist
        profit = self.current_profits[next_loc]
        info = {
            "distance_travelled": distance_travelled,
            "total_distance": self.total_distance,
            "current_profits": self.current_profits,
            "profit": next_loc,
        }
        return (next_state, reward, terminated, truncated, info)

    def _generate_points(self, num_points: int) -> npt.NDArray[np.float32]:
        """Generate random 2D points representing target locations.

        Args:
            num_points (int): Number of points to generate.

        Returns:
            np.ndarray: Array of 2D coordinates for each target.
        """
        return np.random.uniform(low=0, high=self.max_area, size=(num_points, 2)).astype(np.float32)

    def _calculate_distances(self, locations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Calculate the distance matrix between all target locations.

        Args:
            locations: List of 2D target locations.

        Returns:
            np.ndarray: Matrix of pairwise distances between targets.
        """
        n = len(locations)

        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def _get_rewards(self, next_loc: int) -> float:
        """Calculate the reward based on the distance traveled, however if a target gets visited again then it incurs a high penalty.

        Args:
            next_loc (int): Next location of the agent.

        Returns:
            float: Reward based on the travel distance between past and next locations, or negative reward if repeats visit.
        """
        reward = self.current_profits[next_loc] if not self.visited_targets[next_loc] else -1e4
        return float(reward)

    def calculate_soft_upper_max_profit(self) -> float:
        """
        Calculate a soft upper maximum on the profit achievable in an episode.
        This is an optimistic estimate based on the initial configuration.
        """
        # Sort profits in descending order
        sorted_profits = np.sort(self.initial_profits)[::-1]

        # Calculate the minimum spanning tree (MST) distance as a lower bound for travel
        mst_distance = self._calculate_mst_distance()

        # Estimate maximum achievable profit
        max_profit = 0
        remaining_distance = self.max_steps * np.mean(self.distances)  # Assume average distance per step

        for profit in sorted_profits:
            if remaining_distance > 0:
                max_profit += max(0, profit - mst_distance / self.num_targets)
                remaining_distance -= np.mean(self.distances)
            else:
                break

        return max_profit

    def _calculate_mst_distance(self) -> float:
        """
        Calculate the minimum spanning tree distance using Prim's algorithm.
        This provides a lower bound on the distance required to visit all targets.
        """
        n = len(self.locations)
        visited = [False] * n
        min_distances = [float("inf")] * n
        min_distances[0] = 0
        mst_distance = 0

        for _ in range(n):
            v = min((d, i) for i, d in enumerate(min_distances) if not visited[i])[1]
            visited[v] = True
            mst_distance += min_distances[v]

            for u in range(n):
                if not visited[u]:
                    min_distances[u] = min(min_distances[u], self.distances[v][u])

        return mst_distance
