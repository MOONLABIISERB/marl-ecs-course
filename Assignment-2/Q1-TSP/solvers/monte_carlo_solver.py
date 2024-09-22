# solvers/monte_carlo_solver.py

import numpy as np
import random
from typing import List, Tuple, Dict


class MonteCarloSolver:
    """
    Monte Carlo solver for the TSP.
    Learns a policy by simulating episodes and updating value estimates.
    """

    def __init__(self, num_cities: int, distance_matrix: np.ndarray):
        self.num_cities = num_cities
        self.distance_matrix = distance_matrix
        self.Q_values = {}
        self.returns = {}
        self.policy = {}

    def generate_episode(self) -> List[Tuple[Tuple[int, int], int, float]]:
        """Generates an episode by randomly selecting actions."""
        episode = []
        visited = np.zeros(self.num_cities, dtype=bool)
        current_city = random.randint(0, self.num_cities - 1)
        visited[current_city] = True

        while not visited.all():
            possible_actions = [i for i in range(self.num_cities) if not visited[i]]
            action = random.choice(possible_actions)
            reward = -self.distance_matrix[current_city, action]
            state = (current_city, self._state_to_bitmask(visited))
            episode.append((state, action, reward))
            current_city = action
            visited[current_city] = True

        return episode

    def _state_to_bitmask(self, visited: np.ndarray) -> int:
        """Converts the visited cities array to a bitmask."""
        return sum(1 << i for i, v in enumerate(visited) if v)

    def solve(self, num_episodes: int = 10000, method: str = "first_visit"):
        """Trains the policy using Monte Carlo simulations."""
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            visited_state_actions = set()

            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G += reward

                if method == "first_visit" and (state, action) in visited_state_actions:
                    continue

                visited_state_actions.add((state, action))
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(G)
                self.Q_values[(state, action)] = np.mean(self.returns[(state, action)])

                # Update policy
                current_city, _ = state
                possible_actions = [a for a in range(self.num_cities) if a != current_city]
                self.policy[state] = max(
                    possible_actions,
                    key=lambda a: self.Q_values.get((state, a), float('-inf'))
                )

    def get_action(self, current_city: int, visited_cities: np.ndarray) -> int:
        """Selects the next city to visit based on the learned policy."""
        state_bitmask = self._state_to_bitmask(visited_cities)
        state = (current_city, state_bitmask)
        possible_actions = [i for i in range(self.num_cities) if not visited_cities[i]]
        if state in self.policy and self.policy[state] in possible_actions:
            return self.policy[state]
        elif possible_actions:
            return min(possible_actions, key=lambda x: self.distance_matrix[current_city, x])
        else:
            return 0  # Return to starting city if no unvisited cities left
