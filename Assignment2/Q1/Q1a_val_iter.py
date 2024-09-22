import numpy as np
from typing import Dict, List, Optional, Tuple
import gymnasium as gym


class TSP(gym.Env):
    """Traveling Salesman Problem (TSP) RL environment for persistent monitoring."""

    def __init__(self, num_targets: int, max_area: int = 30, seed: int = None) -> None:
        super().__init__()
        if seed is not None:
            np.random.seed(seed=seed)
        self.num_targets = num_targets
        self.max_area = max_area
        self.locations = self._generate_points(num_targets)
        self.distances = self._calculate_distances(self.locations)
        self.visited = None
        self.current_location = None
        self.steps = 0

    def _generate_points(self, num_points: int) -> np.ndarray:
        """Generate random 2D points for targets."""
        points = []
        while len(points) < num_points:
            x = np.random.random() * self.max_area
            y = np.random.random() * self.max_area
            points.append([x, y])
        return np.array(points)

    def _calculate_distances(self, locations: List) -> np.ndarray:
        """Calculate pairwise distances between locations."""
        n = len(locations)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        self.steps = 0
        self.current_location = 0  # Start at the first location
        self.visited = [False] * self.num_targets
        self.visited[self.current_location] = True
        state = (self.current_location, tuple(self.visited))
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Move to the next target (action)."""
        if self.visited[action]:
            return (self.current_location, tuple(self.visited)), -10000, True, False, {}
        reward = -self.distances[self.current_location, action]
        self.visited[action] = True
        self.current_location = action
        self.steps += 1
        done = all(self.visited)
        return (self.current_location, tuple(self.visited)), reward, done, False, {}


def value_iteration(env: TSP, gamma=1.0, max_iterations=1000, tolerance=1e-6):
    """Perform Value Iteration to solve TSP."""
    n = env.num_targets
    state_space_size = (n, 1 << n)  # State space is num_targets * 2^num_targets
    value_function = np.zeros(state_space_size)
    policy = np.zeros(state_space_size, dtype=int)

    for iteration in range(max_iterations):
        delta = 0
        for loc in range(n):
            for visited_set in range(1 << n):
                if (visited_set & (1 << loc)) == 0:
                    continue  # Invalid state (not visited)

                best_value = float('inf')
                best_action = -1

                for next_loc in range(n):
                    if visited_set & (1 << next_loc):
                        continue  # Already visited
                    next_visited_set = visited_set | (1 << next_loc)
                    reward = -env.distances[loc, next_loc]
                    value = reward + gamma * value_function[next_loc, next_visited_set]

                    if value < best_value:
                        best_value = value
                        best_action = next_loc

                if best_action != -1:
                    delta = max(delta, abs(value_function[loc, visited_set] - best_value))
                    value_function[loc, visited_set] = best_value
                    policy[loc, visited_set] = best_action

        if delta < tolerance:
            print(f"Value iteration converged after {iteration} iterations")
            break

    return value_function, policy


def extract_policy(env: TSP, policy):
    """Extract the optimal path from the computed policy."""
    current_city = 0
    visited_set = 1  # Start at city 0, mark it as visited
    path = [current_city]

    while len(path) < env.num_targets:
        next_city = policy[current_city, visited_set]
        path.append(int(next_city))  # Convert np.int64 to regular int
        visited_set |= (1 << next_city)
        current_city = next_city

    return path


if __name__ == "__main__":
    # Initialize the environment with 6 targets
    num_targets = 5
    env = TSP(num_targets)

    # Perform value iteration
    value_function, policy = value_iteration(env)

    # Extract and print the optimal path
    optimal_path = extract_policy(env, policy)
    print(f"Optimal Path: {optimal_path}")

    # Print state values and the optimum policy after convergence
    print("\nOptimum State Values and Policy after Convergence:")
    for loc in range(num_targets):
        for visited_set in range(1 << num_targets):
            if (visited_set & (1 << loc)) != 0:  # Only print valid states
                state_value = value_function[loc, visited_set]
                optimal_next_city = policy[loc, visited_set]
                print(f"State (loc={loc}, visited_set={bin(visited_set)}): Value={state_value:.4f}, "
                      f"Next City={optimal_next_city}")
