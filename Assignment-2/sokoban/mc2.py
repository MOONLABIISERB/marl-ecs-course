import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sokoban import SokobanEnv


class MonteCarlo:
    def __init__(self, env):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.policy = {}
        self.initial_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor

    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return (
                np.argmax(self.q_values[state])
                if state in self.q_values
                else self.env.action_space.sample()
            )

    def generate_episode(self, epsilon):
        episode = []
        state, _ = self.env.reset()
        done = False
        steps = 0
        while not done and steps < 100:  # Limit episode length
            state = self.env.get_state()
            action = self.epsilon_greedy_policy(state, epsilon)
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            steps += 1
        return episode

    def train(self, num_episodes=10000):
        epsilon = self.initial_epsilon
        pbar = tqdm(range(num_episodes), desc="MC training")
        for _ in pbar:
            episode = self.generate_episode(epsilon)
            G = 0
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if (state, action) not in [(x[0], x[1]) for x in episode[0:t]]:
                    self.returns[state][action].append(G)
                    self.q_values[state][action] = np.mean(self.returns[state][action])
                    self.policy[state] = np.argmax(self.q_values[state])

            # Decay epsilon
            epsilon = max(self.min_epsilon, epsilon * self.epsilon_decay)

            # Update progress bar
            pbar.set_postfix({"epsilon": f"{epsilon:.4f}"})

        return self.policy

    def get_action(self, state):
        return self.policy.get(state, self.env.action_space.sample())
