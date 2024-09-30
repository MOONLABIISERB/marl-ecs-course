from tsp import TSP
from typing import Tuple
import numpy as np

def solve_tsp(env: TSP, gamma: float = 1.0, max_iter: int = 1000, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the Traveling Salesman Problem using Value Iteration."""
    n_targets = env.num_targets
    state_shape = (n_targets, 1 << n_targets)
    values = np.zeros(state_shape)
    policy = np.zeros(state_shape, dtype=int)

    for iteration in range(max_iter):
        max_delta = 0
        for current in range(n_targets):
            for visited in range(1 << n_targets):
                if not (visited & (1 << current)):
                    continue  # Skip invalid states

                min_value = float('inf')
                best_action = -1
                for next_target in range(n_targets):
                    if visited & (1 << next_target):
                        continue  # Skip already visited targets

                    new_visited = visited | (1 << next_target)
                    cost = env.distances[current, next_target]
                    total_value = -cost + gamma * values[next_target, new_visited]

                    if total_value < min_value:
                        min_value = total_value
                        best_action = next_target

                if best_action != -1:
                    max_delta = max(max_delta, abs(values[current, visited] - min_value))
                    values[current, visited] = min_value
                    policy[current, visited] = best_action

        if max_delta < tol:
            print(f"Converged after {iteration + 1} iterations")
            break

    return values, policy

def get_optimal_path(env: TSP, policy: np.ndarray) -> list:
    """Extract the optimal path from the computed policy."""
    path = [0]  # Start at city 0
    current = 0
    visited = 1  # Binary representation of visited cities

    while len(path) < env.num_targets:
        next_city = policy[current, visited]
        path.append(int(next_city))
        visited |= (1 << next_city)
        current = next_city

    return path

def main():
    # Set up the TSP environment
    n_targets = 5
    env = TSP(n_targets)

    # Solve the TSP
    values, policy = solve_tsp(env)

    # Get and print the optimal path
    optimal_path = get_optimal_path(env, policy)
    print(f"Optimal Path: {optimal_path}")

    # Print state values and optimal policy
    print("\nState Values and Optimal Policy:")
    for loc in range(n_targets):
        for visited in range(1 << n_targets):
            if visited & (1 << loc):
                value = values[loc, visited]
                next_city = policy[loc, visited]
                print(f"State (loc={loc}, visited={visited}): Value={value:.4f}, Next={next_city}")

if __name__ == "__main__":
    main()