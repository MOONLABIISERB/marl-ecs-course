# solvers/dynamic_programming_solver.py

import numpy as np
from typing import List, Dict, Tuple


class DynamicProgrammingSolver:
    """
    Dynamic Programming solver for the TSP.
    Computes the shortest path visiting all cities exactly once.
    """

    def __init__(self, num_cities: int, distance_matrix: np.ndarray):
        self.num_cities = num_cities
        self.distance_matrix = distance_matrix
        self.value_table = {}
        self.policy_table = {}
        self.total_min_cost = None
        self.last_city = None

    def solve(self):
        """Solves the TSP using dynamic programming."""
        num_subsets = 1 << self.num_cities  # Total number of subsets

        # Initialize the value table with infinity
        for subset in range(num_subsets):
            for last in range(self.num_cities):
                self.value_table[(subset, last)] = np.inf
        self.value_table[(1 << 0, 0)] = 0  # Starting from city 0

        # Dynamic programming to fill the value table
        for subset_size in range(2, self.num_cities + 1):
            for subset in [s for s in range(num_subsets) if bin(s).count('1') == subset_size and s & (1 << 0)]:
                for last in range(self.num_cities):
                    if not (subset & (1 << last)):
                        continue
                    prev_subset = subset ^ (1 << last)
                    if prev_subset == 0:
                        continue
                    for k in range(self.num_cities):
                        if (prev_subset & (1 << k)):
                            cost = self.value_table[(prev_subset, k)] + self.distance_matrix[k, last]
                            if cost < self.value_table[(subset, last)]:
                                self.value_table[(subset, last)] = cost
                                self.policy_table[(subset, last)] = k

        # Close the tour by returning to the starting city
        full_subset = (1 << self.num_cities) - 1
        min_cost = np.inf
        last_city = None
        for k in range(self.num_cities):
            if k == 0:
                continue
            cost = self.value_table[(full_subset, k)] + self.distance_matrix[k, 0]
            if cost < min_cost:
                min_cost = cost
                last_city = k
        self.total_min_cost = min_cost
        self.last_city = last_city

    def reconstruct_path(self) -> List[int]:
        """Reconstructs the optimal path from the computed policy."""
        subset = (1 << self.num_cities) - 1
        path = [0]
        last_city = self.last_city
        for _ in range(self.num_cities - 1):
            path.append(last_city)
            subset ^= (1 << last_city)
            last_city = self.policy_table.get((subset | (1 << last_city), last_city))
            if last_city is None:
                break
        path.append(0)  # Return to the starting city
        path.reverse()
        return path

    def get_action(self, current_city: int, visited_cities: np.ndarray) -> int:
        """Determines the next city to visit based on the policy."""
        subset = sum(1 << i for i in range(self.num_cities) if visited_cities[i])
        if subset == (1 << self.num_cities) - 1:
            return 0  # Return to starting city if all visited

        min_cost = np.inf
        next_city = None
        for city in range(self.num_cities):
            if not visited_cities[city]:
                s = subset | (1 << city)
                if (s, city) in self.value_table:
                    cost = self.value_table[(s, city)] + self.distance_matrix[city, 0]
                    if cost < min_cost:
                        min_cost = cost
                        next_city = city
        if next_city is not None:
            return next_city
        else:
            # If policy is undefined, select the nearest unvisited city
            unvisited = [i for i in range(self.num_cities) if not visited_cities[i]]
            return min(unvisited, key=lambda x: self.distance_matrix[current_city, x])
