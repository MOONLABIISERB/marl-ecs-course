import numpy as np
from typing import List
import random


class DynamicProgrammingTSP:
    def __init__(self, total_targets: int, distance_matrix: np.ndarray):
        self.total_targets = total_targets
        self.distance_matrix = distance_matrix
        self.value_function = {}
        self.optimal_policy = {}

    def compute_solution(self, max_iter: int = 1000, tolerance: float = 1e-6):
        """Use value iteration to solve the TSP problem."""
        for visited_mask in range(1 << self.total_targets):
            for last_target in range(self.total_targets):
                if (visited_mask & (1 << last_target)) != 0:
                    self.value_function[(visited_mask, last_target)] = float("inf")
        self.value_function[(1, 0)] = 0  # Starting at target 0

        # Perform value iteration
        for _ in range(max_iter):
            max_change = 0
            for visited_mask in range(1, 1 << self.total_targets):
                for last_target in range(self.total_targets):
                    if (visited_mask & (1 << last_target)) == 0:
                        continue
                    previous_value = self.value_function.get((visited_mask, last_target), float("inf"))

                    # Calculate potential next states
                    next_values = [
                        self.distance_matrix[last_target][next_target]
                        + self.value_function.get((visited_mask & ~(1 << last_target), next_target), float("inf"))
                        for next_target in range(self.total_targets)
                        if next_target != last_target and (visited_mask & (1 << next_target))
                    ]

                    if next_values:
                        self.value_function[(visited_mask, last_target)] = min(next_values)
                        max_change = max(max_change, abs(previous_value - self.value_function[(visited_mask, last_target)]))

            if max_change < tolerance:
                break

        # Build the optimal policy
        for visited_mask in range(1, 1 << self.total_targets):
            for last_target in range(self.total_targets):
                if (visited_mask & (1 << last_target)) == 0:
                    continue
                next_states = [
                    (
                        self.distance_matrix[last_target][next_target]
                        + self.value_function.get((visited_mask & ~(1 << last_target), next_target), float("inf")),
                        next_target,
                    )
                    for next_target in range(self.total_targets)
                    if next_target != last_target and (visited_mask & (1 << next_target))
                ]
                if next_states:
                    self.optimal_policy[(visited_mask, last_target)] = min(next_states)[1]

    def select_action(self, current_position: int, visited_targets: List[int]) -> int:
        """Return the best action (next target) based on the current state."""
        visited_mask = sum(1 << i for i in visited_targets)
        if (visited_mask, current_position) in self.optimal_policy:
            return self.optimal_policy[(visited_mask, current_position)]
        unvisited_targets = [i for i in range(self.total_targets) if i not in visited_targets]
        return min(unvisited_targets, key=lambda x: self.distance_matrix[current_position][x])


class MonteCarloTSP:
    def __init__(self, total_targets: int, distance_matrix: np.ndarray):
        self.total_targets = total_targets
        self.distance_matrix = distance_matrix
        self.Q_values = {}
        self.returns = {}
        self.policy = {}

    def simulate_episode(self):
        """Generate a random episode for Monte Carlo estimation."""
        episode = []
        visited_set = set()
        current_target = random.randint(0, self.total_targets - 1)
        visited_set.add(current_target)

        while len(visited_set) < self.total_targets:
            available_targets = [i for i in range(self.total_targets) if i not in visited_set]
            next_target = random.choice(available_targets)
            travel_reward = -self.distance_matrix[current_target][next_target]
            episode.append((current_target, next_target, travel_reward))
            current_target = next_target
            visited_set.add(current_target)

        return episode

    def compute_solution(self, episodes: int, method: str = "first_visit"):
        """Solve using Monte Carlo method with first visit or every visit."""
        for _ in range(episodes):
            episode = self.simulate_episode()
            total_reward = 0
            visited_state_action_pairs = set()

            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                total_reward += reward

                if method == "first_visit" and (state, action) in visited_state_action_pairs:
                    continue

                visited_state_action_pairs.add((state, action))
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(total_reward)
                self.Q_values[(state, action)] = np.mean(self.returns[(state, action)])

                # Update policy based on Q-values
                if state not in self.policy or self.Q_values[(state, action)] > self.Q_values.get((state, self.policy.get(state)), float("-inf")):
                    self.policy[state] = action

    def select_action(self, current_position: int, visited_targets: List[int]) -> int:
        """Select the best action or next target to visit."""
        if current_position in self.policy:
            return self.policy[current_position]
        unvisited_targets = [i for i in range(self.total_targets) if i not in visited_targets]
        return min(unvisited_targets, key=lambda x: self.distance_matrix[current_position][x])


class EpsilonGreedyMonteCarloTSP:
    def __init__(self, total_targets: int, distance_matrix: np.ndarray, epsilon: float = 0.1):
        self.total_targets = total_targets
        self.distance_matrix = distance_matrix
        self.Q_values = {}
        self.returns = {}
        self.policy = {}
        self.epsilon = epsilon

    def epsilon_greedy_selection(self, state: int, available_targets: List[int]) -> int:
        """Choose an action using an epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(available_targets)
        else:
            q_values = [self.Q_values.get((state, target), 0) for target in available_targets]
            return available_targets[np.argmax(q_values)]

    def simulate_episode(self):
        """Generate an episode using epsilon-greedy policy."""
        episode = []
        visited_set = set()
        current_target = random.randint(0, self.total_targets - 1)
        visited_set.add(current_target)

        while len(visited_set) < self.total_targets:
            available_targets = [i for i in range(self.total_targets) if i not in visited_set]
            next_target = self.epsilon_greedy_selection(current_target, available_targets)
            travel_reward = -self.distance_matrix[current_target][next_target]
            episode.append((current_target, next_target, travel_reward))
            current_target = next_target
            visited_set.add(current_target)

        return episode

    def compute_solution(self, episodes: int, method: str = "first_visit"):
        """Solve using epsilon-greedy Monte Carlo."""
        for _ in range(episodes):
            episode = self.simulate_episode()
            total_reward = 0
            visited_state_action_pairs = set()

            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                total_reward += reward

                if method == "first_visit" and (state, action) in visited_state_action_pairs:
                    continue

                visited_state_action_pairs.add((state, action))
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(total_reward)
                self.Q_values[(state, action)] = np.mean(self.returns[(state, action)])

                available_targets = [i for i in range(self.total_targets) if i != state]
                best_action = max(available_targets, key=lambda a: self.Q_values.get((state, a), 0))
                self.policy[state] = best_action

    def select_action(self, current_position: int, visited_targets: List[int]) -> int:
        """Select the best action using epsilon-greedy policy."""
        available_targets = [i for i in range(self.total_targets) if i not in visited_targets]
        if not available_targets:
            return current_position  # Return to start if all targets visited
        return self.epsilon_greedy_selection(current_position, available_targets)


def initialize_dp_solver(env):
    """Initialize and solve TSP using dynamic programming."""
    solver = DynamicProgrammingTSP(env.total_targets, env.distance_matrix)
    solver.compute_solution()
    return solver


def initialize_mc_solver(env, episodes=10000, method="first_visit"):
    """Initialize and solve TSP using Monte Carlo method."""
    solver = MonteCarloTSP(env.total_targets, env.distance_matrix)
    solver.compute_solution(episodes, method)
    return solver


def initialize_mc_epsilon_greedy_solver(env, episodes=1000000, method="first_visit", epsilon=0.1):
    """Initialize and solve TSP using epsilon-greedy Monte Carlo method."""
    solver = EpsilonGreedyMonteCarloTSP(env.total_targets, env.distance_matrix, epsilon)
    solver.compute_solution(episodes, method)
    return solver
