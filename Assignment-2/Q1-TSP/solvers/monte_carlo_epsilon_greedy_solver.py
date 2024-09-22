# solvers/monte_carlo_epsilon_greedy_solver.py

import numpy as np
import random
from typing import List, Tuple, Dict


class MonteCarloEpsilonGreedySolver:
    """
    Monte Carlo solver with epsilon-greedy policy for the TSP.
    Balances exploration and exploitation during policy learning.
    """

    def __init__(self, num_cities: int, distance_matrix: np.ndarray, epsilon: float = 0.1):
        self.num_cities = num_cities
        self.distance_matrix = distance_matrix
        self.epsilon = epsilon
        self.Q_values = {}
        self.returns = {}
        self.policy = {}

    def epsilon_greedy(self, state: Tuple[int, int], possible_actions: List[int]) -> int:
        """Selects an action using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            q_values = [(action, self.Q_values.get((state, action), float('-inf'))) for action in possible_actions]
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_actions = [action for action, q in q_values if q == max_q]
            return random.choice(best_actions)

    def generate_episode(self) -> List[Tuple[Tuple[int, int], int, float]]:
        """Generates an episode using the epsilon-greedy policy."""
        episode = []
        visited = np.zeros(self.num_cities, dtype=bool)
        current_city = random.randint(0, self.num_cities - 1)
        visited[current_city] = True

        while not visited.all():
            possible_actions = [i for i in range(self.num_cities) if not visited[i]]
            state_bitmask = self._state_to_bitmask(visited)
            state = (current_city, state_bitmask)
            action = self.epsilon_greedy(state, possible_actions)
            reward = -self.distance_matrix[current_city, action]
            episode.append((state, action, reward))
            current_city = action
            visited[current_city] = True

        return episode

    def _state_to_bitmask(self, visited: np.ndarray) -> int:
        """Converts the visited cities array to a bitmask."""
        return sum(1 << i for i, v in enumerate(visited) if v)

    def solve(self, num_episodes: int = 10000, method: str = "first_visit"):
        """Trains the policy using Monte Carlo simulations with epsilon-greedy exploration."""
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
        """Selects the next city to visit using the epsilon-greedy policy."""
        possible_actions = [i for i in range(self.num_cities) if not visited_cities[i]]
        if not possible_actions:
            return 0  # Return to starting city if all cities are visited

        state_bitmask = self._state_to_bitmask(visited_cities)
        state = (current_city, state_bitmask)
        return self.epsilon_greedy(state, possible_actions)
