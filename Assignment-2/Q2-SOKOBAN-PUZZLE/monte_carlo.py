# monte_carlo.py

import numpy as np
from collections import defaultdict
from tqdm import tqdm


class MonteCarloAgent:
    """
    Agent that uses Monte Carlo methods for policy computation.
    """

    def __init__(self, env):
        self.env = env
        self.Q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.policy = {}
        self.initial_epsilon = 1.0
        self.minimum_epsilon = 0.01
        self.epsilon_decay_rate = 0.995
        self.discount_factor = 0.99

    def _epsilon_greedy(self, state, epsilon):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_values[state]) if state in self.Q_values else self.env.action_space.sample()

    def generate_episode(self, epsilon):
        # Generate an episode using the current policy
        episode = []
        self.env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            state = self.env.get_state()
            action = self._epsilon_greedy(state, epsilon)
            _, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            steps += 1
        return episode

    def train(self, num_episodes=10000):
        # Train the agent using Monte Carlo method
        epsilon = self.initial_epsilon
        for _ in tqdm(range(num_episodes), desc="Monte Carlo Training Progress"):
            episode = self.generate_episode(epsilon)
            G = 0
            visited_state_actions = set()
            for state, action, reward in reversed(episode):
                G = self.discount_factor * G + reward
                if (state, action) not in visited_state_actions:
                    visited_state_actions.add((state, action))
                    self.returns[state][action].append(G)
                    self.Q_values[state][action] = np.mean(self.returns[state][action])
                    self.policy[state] = np.argmax(self.Q_values[state])
            epsilon = max(self.minimum_epsilon, epsilon * self.epsilon_decay_rate)

    def select_action(self, state):
        # Select action based on the learned policy
        return self.policy.get(state, self.env.action_space.sample())
