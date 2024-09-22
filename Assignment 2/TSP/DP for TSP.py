import numpy as np
from typing import Dict, Tuple

class TSP:
    """Traveling Salesman Problem (TSP) RL environment for persistent monitoring."""

    def __init__(self, num_targets: int, max_area: int = 30, seed: int = None) -> None:
        if seed is not None:
            np.random.seed(seed=seed)

        self.num_targets: int = num_targets
        self.max_area: int = max_area

        # Generate random points as target locations
        self.locations: np.ndarray = self._generate_points(self.num_targets)
        # Calculate distances between each pair of locations
        self.distances: np.ndarray = self._calculate_distances(self.locations)

    def _generate_points(self, num_points: int) -> np.ndarray:
        """Generate random 2D points representing target locations."""
        points = []
        while len(points) < num_points:
            x = np.random.random() * self.max_area
            y = np.random.random() * self.max_area
            if [x, y] not in points:
                points.append([x, y])
        return np.array(points)

    def _calculate_distances(self, locations: np.ndarray) -> np.ndarray:
        """Calculate the distance matrix between all target locations."""
        n = len(locations)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(locations[i] - locations[j])
        return distances

    def solve_tsp_value_iteration(self, gamma: float = 0.9, max_iterations: int = 10000, theta: float = 1e-6) -> Tuple[Dict, np.ndarray]:
        """Solve TSP using Value Iteration.
        
        Args:
            gamma (float): Discount factor for future rewards.
            max_iterations (int): Maximum number of iterations for value iteration.
            theta (float): Convergence threshold.

        Returns:
            Tuple[Dict, np.ndarray]: Optimal policy and the value function.
        """
        # Number of states: (location, visited_mask)
        V = np.zeros((2 ** self.num_targets, self.num_targets))  # Value function
        policy = {}

        for iteration in range(max_iterations):
            delta = 0
            for visited_mask in range(2 ** self.num_targets):
                for loc in range(self.num_targets):
                    if visited_mask & (1 << loc):  # Skip if location is already visited
                        continue

                    v = V[visited_mask][loc]  # Previous value for the state
                    best_value = float('inf')
                    
                    # Find the best action (next location to visit)
                    for next_loc in range(self.num_targets):
                        if visited_mask & (1 << next_loc):  # Skip already visited locations
                            continue
                        
                        # Compute next state
                        next_visited_mask = visited_mask | (1 << next_loc)
                        next_value = -self.distances[loc][next_loc] + gamma * V[next_visited_mask][next_loc]
                        
                        best_value = min(best_value, next_value)  # Minimize the distance (cost)
                    
                    V[visited_mask][loc] = best_value  # Update value function
                    delta = max(delta, abs(v - best_value))  # Track largest change in value function

            if delta < theta:  # Stop if the value function has converged
                print(f"Value Iteration converged after {iteration + 1} iterations.")
                break

        # Extract policy from value function
        for visited_mask in range(2 ** self.num_targets):
            for loc in range(self.num_targets):
                if visited_mask & (1 << loc):  # If already visited, no action
                    continue
                
                best_action = None
                best_value = float('inf')
                
                # Choose the best next action
                for next_loc in range(self.num_targets):
                    if visited_mask & (1 << next_loc):
                        continue
                    
                    next_visited_mask = visited_mask | (1 << next_loc)
                    next_value = -self.distances[loc][next_loc] + gamma * V[next_visited_mask][next_loc]
                    
                    if next_value < best_value:
                        best_value = next_value
                        best_action = next_loc
                
                policy[(visited_mask, loc)] = best_action  # Best action for (visited_mask, loc)

        return policy, V

if __name__ == "__main__":
    num_targets = 6
    env = TSP(num_targets)

    # Solve TSP using value iteration
    optimal_policy, value_function = env.solve_tsp_value_iteration()

    print("Optimal Policy:", optimal_policy)

    # Testing the learned policy
    current_loc = 0
    visited_mask = 0
    total_reward = 0

    print("\nTesting the learned policy:")
    while visited_mask != (1 << num_targets) - 1:  # Until all locations are visited
        try:
            action = optimal_policy[(visited_mask, current_loc)]
            if action is None:
                raise KeyError
        except KeyError:
            # If the state isn't in the policy, choose the closest unvisited location
            unvisited_locs = [loc for loc in range(num_targets) if not (visited_mask & (1 << loc))]
            action = min(unvisited_locs, key=lambda loc: env.distances[current_loc][loc])

        print(f"At location {current_loc}, going to {action}")
        
        visited_mask |= (1 << action)  # Mark this location as visited
        total_reward += -env.distances[current_loc][action]  # Add the negative of the distance (cost)
        current_loc = action  # Move to the next location
    
    print(f"Total reward (negative of total distance): {total_reward}")
