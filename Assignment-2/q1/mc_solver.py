import numpy as np
from collections import defaultdict
from tsp import TSP

class MCSolver:
    def __init__(self, env: TSP, epsilon: float = 0.1, discount_factor: float = 0.9):
        self.env = env
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.q_table = defaultdict(lambda: np.zeros(env.num_targets))
        self.returns = defaultdict(list)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def train(self, num_episodes: int = 1000):
        for episode in range(num_episodes):
            episode_history = []
            state, _ = self.env.reset()
            state = int(state[0])
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_history.append((state, action, reward))
                state = int(next_state[0])

            # Calculate returns and update Q-values
            G = 0
            for state, action, reward in reversed(episode_history):
                G = reward + self.discount_factor * G
                self.returns[(state, action)].append(G)
                self.q_table[state][action] = np.mean(self.returns[(state, action)])

            if episode % 100 == 0:
                print(f"Episode {episode} completed")

    def get_optimal_policy(self):
        return {state: np.argmax(actions) for state, actions in self.q_table.items()}

if __name__ == "__main__":
    num_targets = 50
    env = TSP(num_targets)
    solver = MCSolver(env)

    print("Training MC solver...")
    solver.train()

    optimal_policy = solver.get_optimal_policy()
    print("Optimal policy:", optimal_policy)

    # Evaluate the policy
    state, _ = env.reset()
    total_reward = 0
    for _ in range(100):
        action = optimal_policy.get(int(state[0]), env.action_space.sample())
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"Total reward using optimal policy: {total_reward}")
