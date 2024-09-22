import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sokoban import SokobanEnv


class MonteCarloSolver:
    """Monte Carlo Solver for Sokoban using an epsilon-greedy policy."""

    def __init__(self, env):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns_table = defaultdict(lambda: defaultdict(list))
        self.policy_table = {}
        self.start_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay_factor = 0.995
        self.discount_factor = 0.99  # Gamma, the discount factor for future rewards

    def select_action_epsilon_greedy(self, state, epsilon):
        """Select an action using epsilon-greedy strategy."""
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            return (
                np.argmax(self.q_table[state])
                if state in self.q_table
                else self.env.action_space.sample()
            )

    def play_episode(self, epsilon):
        """Simulate an episode following the epsilon-greedy policy."""
        episode = []
        state, _ = self.env.reset()
        done = False
        step_count = 0

        while not done and step_count < 100:  # Limit to 100 steps per episode
            state = self.env.get_state()
            action = self.select_action_epsilon_greedy(state, epsilon)
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            step_count += 1

        return episode

    def train_model(self, episodes=10000):
        """Train the Monte Carlo Solver using multiple episodes."""
        epsilon = self.start_epsilon
        pbar = tqdm(range(episodes), desc="Monte Carlo Training")

        for _ in pbar:
            episode = self.play_episode(epsilon)
            cumulative_return = 0  # This represents the return G_t

            # Loop through episode backwards
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                cumulative_return = self.discount_factor * cumulative_return + reward

                if (state, action) not in [(step[0], step[1]) for step in episode[0:t]]:
                    self.returns_table[state][action].append(cumulative_return)
                    self.q_table[state][action] = np.mean(self.returns_table[state][action])
                    self.policy_table[state] = np.argmax(self.q_table[state])

            # Epsilon decay after each episode to reduce exploration over time
            epsilon = max(self.min_epsilon, epsilon * self.epsilon_decay_factor)

            # Update progress bar with current epsilon value
            pbar.set_postfix({"epsilon": f"{epsilon:.4f}"})

        return self.policy_table

    def choose_action(self, state):
        """Get the best action based on the learned policy."""
        return self.policy_table.get(state, self.env.action_space.sample())
