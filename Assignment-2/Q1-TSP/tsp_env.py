# tsp_env.py

import numpy as np
import gym
from gym import spaces
from typing import Optional, Tuple, List, Dict


class TSPEnv(gym.Env):
    """
    Custom OpenAI Gym environment for the Traveling Salesman Problem (TSP).
    The goal is to find the shortest possible route that visits each city exactly once.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, num_cities: int, area_size: int = 30, seed: Optional[int] = None):
        super(TSPEnv, self).__init__()
        self.num_cities = num_cities
        self.area_size = area_size
        self.seed(seed)

        # Generate random positions for the cities
        self.city_positions = self._generate_city_positions()

        # Calculate the distance matrix
        self.distance_matrix = self._calculate_distance_matrix()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.num_cities)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.num_cities),  # Current city
            spaces.MultiBinary(self.num_cities)  # Visited cities
        ))

        self.current_city = 0
        self.visited_cities = np.zeros(self.num_cities, dtype=np.int8)
        self.total_distance = 0.0

    def _generate_city_positions(self) -> np.ndarray:
        """Generates random (x, y) positions for each city within the specified area."""
        return np.random.uniform(0, self.area_size, size=(self.num_cities, 2))

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculates the Euclidean distance between all pairs of cities."""
        num = self.num_cities
        dist_matrix = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                if i != j:
                    dist_matrix[i, j] = np.linalg.norm(self.city_positions[i] - self.city_positions[j])
        return dist_matrix

    def reset(self) -> Tuple[int, np.ndarray]:
        """Resets the environment to the initial state."""
        self.current_city = 0
        self.visited_cities = np.zeros(self.num_cities, dtype=np.int8)
        self.visited_cities[self.current_city] = 1
        self.total_distance = 0.0
        return self.current_city, self.visited_cities.copy()

    def step(self, action: int) -> Tuple[Tuple[int, np.ndarray], float, bool, bool, Dict]:
        """Performs the action of moving to the next city."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        done = False

        if self.visited_cities[action]:
            # High penalty for revisiting a city
            reward = -10000.0
            done = True  # End the episode on invalid action
        else:
            # Calculate distance to the next city
            distance = self.distance_matrix[self.current_city, action]
            self.total_distance += distance
            reward = -distance  # Negative reward for traveling distance
            self.current_city = action
            self.visited_cities[action] = 1

            # Check if all cities have been visited
            if self.visited_cities.sum() == self.num_cities:
                done = True

        observation = (self.current_city, self.visited_cities.copy())
        info = {}

        return observation, reward, done, False, info

    def render(self, mode='human'):
        """Renders the environment (not implemented)."""
        pass

    def seed(self, seed: Optional[int] = None):
        """Sets the random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
