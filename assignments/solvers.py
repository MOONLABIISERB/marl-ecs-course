import numpy as np
from typing import List, Tuple
import random


class TSPDynamicProgramming:
    def __init__(self, num_targets: int, distances: np.ndarray):
        self.num_targets = num_targets
        self.distances = distances
        self.V = {}
        self.policy = {}

    def solve(self, max_iterations: int = 1000, threshold: float = 1e-6):
        # Initialize value function
        for mask in range(1 << self.num_targets):
            for last in range(self.num_targets):
                if (mask & (1 << last)) != 0:
                    self.V[(mask, last)] = float("inf")
        self.V[(1, 0)] = 0  # Start at node 0

        # Value iteration
        for _ in range(max_iterations):
            delta = 0
            for mask in range(1, 1 << self.num_targets):
                for last in range(self.num_targets):
                    if (mask & (1 << last)) == 0:
                        continue
                    old_v = self.V.get((mask, last), float("inf"))

                    # Calculate values for all possible next states
                    next_values = [
                        self.distances[last][j]
                        + self.V.get((mask & ~(1 << last), j), float("inf"))
                        for j in range(self.num_targets)
                        if j != last and (mask & (1 << j))
                    ]

                    # Update V only if there are valid next states
                    if next_values:
                        self.V[(mask, last)] = min(next_values)
                        delta = max(delta, abs(old_v - self.V[(mask, last)]))

            if delta < threshold:
                break

        # Extract policy
        for mask in range(1, 1 << self.num_targets):
            for last in range(self.num_targets):
                if (mask & (1 << last)) == 0:
                    continue
                next_states = [
                    (
                        self.distances[last][j]
                        + self.V.get((mask & ~(1 << last), j), float("inf")),
                        j,
                    )
                    for j in range(self.num_targets)
                    if j != last and (mask & (1 << j))
                ]
                if next_states:
                    self.policy[(mask, last)] = min(next_states)[1]

    def get_action(self, current_state: int, visited: List[int]) -> int:
        mask = sum(1 << i for i in visited)
        if (mask, current_state) in self.policy:
            return self.policy[(mask, current_state)]
        # If policy is not defined, choose the nearest unvisited target
        unvisited = [i for i in range(self.num_targets) if i not in visited]
        return min(unvisited, key=lambda x: self.distances[current_state][x])


class TSPMonteCarlo:
    def __init__(self, num_targets: int, distances: np.ndarray):
        self.num_targets = num_targets
        self.distances = distances
        self.Q = {}
        self.returns = {}
        self.policy = {}

    def generate_episode(self):
        episode = []
        visited = set()
        current = random.randint(0, self.num_targets - 1)
        visited.add(current)

        while len(visited) < self.num_targets:
            available_actions = [i for i in range(self.num_targets) if i not in visited]
            action = random.choice(available_actions)
            reward = -self.distances[current][action]
            episode.append((current, action, reward))
            current = action
            visited.add(current)

        return episode

    def solve(self, num_episodes: int, method: str = "first_visit"):
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            visited_sa = set()

            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G += reward

                if method == "first_visit" and (state, action) in visited_sa:
                    continue

                visited_sa.add((state, action))
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(G)
                self.Q[(state, action)] = np.mean(self.returns[(state, action)])

                # Update policy
                if state not in self.policy or self.Q[(state, action)] > self.Q.get(
                    (state, self.policy.get(state)), float("-inf")
                ):
                    self.policy[state] = action

    def get_action(self, current_state: int, visited: List[int]) -> int:
        if current_state in self.policy:
            return self.policy[current_state]
        # If policy is not defined, choose the nearest unvisited target
        unvisited = [i for i in range(self.num_targets) if i not in visited]
        return min(unvisited, key=lambda x: self.distances[current_state][x])


class TSPMonteCarloEpsilonGreedy:
    def __init__(self, num_targets: int, distances: np.ndarray, epsilon: float = 0.1):
        self.num_targets = num_targets
        self.distances = distances
        self.Q = {}  # Action-value function
        self.returns = {}  # Returns for each state-action pair
        self.policy = {}  # Policy (best action for each state)
        self.epsilon = epsilon  # Exploration rate

    def epsilon_greedy_action(self, state: int, available_actions: List[int]) -> int:
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            q_values = [self.Q.get((state, a), 0) for a in available_actions]
            return available_actions[np.argmax(q_values)]

    def generate_episode(self):
        episode = []
        visited = set()
        current = random.randint(0, self.num_targets - 1)
        visited.add(current)

        while len(visited) < self.num_targets:
            available_actions = [i for i in range(self.num_targets) if i not in visited]
            action = self.epsilon_greedy_action(current, available_actions)
            reward = -self.distances[current][action]
            episode.append((current, action, reward))
            current = action
            visited.add(current)

        return episode

    def solve(self, num_episodes: int, method: str = "first_visit"):
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            visited_sa = set()

            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G += reward

                if method == "first_visit" and (state, action) in visited_sa:
                    continue

                visited_sa.add((state, action))
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(G)
                self.Q[(state, action)] = np.mean(self.returns[(state, action)])

                # Update policy
                available_actions = [i for i in range(self.num_targets) if i != state]
                best_action = max(
                    available_actions, key=lambda a: self.Q.get((state, a), 0)
                )
                self.policy[state] = best_action

    def get_action(self, current_state: int, visited: List[int]) -> int:
        available_actions = [i for i in range(self.num_targets) if i not in visited]
        if not available_actions:
            return current_state  # Return to starting point if all targets visited
        return self.epsilon_greedy_action(current_state, available_actions)


def create_dp_solver(env):
    solver = TSPDynamicProgramming(env.num_targets, env.distances)
    solver.solve()
    return solver


def create_mc_solver(env, num_episodes=10000, method="first_visit"):
    solver = TSPMonteCarlo(env.num_targets, env.distances)
    solver.solve(num_episodes, method)
    return solver


def create_mc_epsilon_greedy_solver(
    env, num_episodes=1000000, method="first_visit", epsilon=0.1
):
    solver = TSPMonteCarloEpsilonGreedy(env.num_targets, env.distances, epsilon)
    solver.solve(num_episodes, method)
    return solver
