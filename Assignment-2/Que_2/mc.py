import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sokoban import SokobanGame

class MonteCarlo:
    def __init__(self, environment):
        self.env = environment
        self.q_values = defaultdict(lambda: np.zeros(environment.action_space.n))
        self.state_action_returns = defaultdict(lambda: defaultdict(list))
        self.policy = {}
        self.initial_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay_rate = 0.95
        self.discount_factor = 0.9  # Gamma (discount factor)

    def epsilon_greedy(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values[state]) if state in self.q_values else self.env.action_space.sample()

    def create_episode(self, epsilon):
        episode = []
        state, _ = self.env.reset()
        done = False
        step_count = 0
        while not done and step_count < 100:  # Limiting episode length
            state = self.env.get_current_state()
            action = self.epsilon_greedy(state, epsilon)
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            step_count += 1
        return episode

    def learn(self, num_episodes=10000):
        epsilon = self.initial_epsilon
        progress_bar = tqdm(range(num_episodes), desc="Monte Carlo Training")
        for _ in progress_bar:
            episode = self.create_episode(epsilon)
            G = 0
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.discount_factor * G + reward
                if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                    self.state_action_returns[state][action].append(G)
                    self.q_values[state][action] = np.mean(self.state_action_returns[state][action])
                    self.policy[state] = np.argmax(self.q_values[state])

            # Decay epsilon
            epsilon = max(self.min_epsilon, epsilon * self.epsilon_decay_rate)

            # Update progress bar
            progress_bar.set_postfix({"epsilon": f"{epsilon:.4f}"})

        return self.policy

    def select_action(self, state):
        return self.policy.get(state, self.env.action_space.sample())
