import numpy as np
from tsp import TSP

class DPSolver:
    def __init__(self, env: TSP, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((env.num_targets, env.num_targets))

    def train(self, num_episodes: int = 1000):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            current_target = int(state[0])
            done = False

            while not done:
                action = np.argmax(self.q_table[current_target])
                next_state, reward, done, _, _ = self.env.step(action)
                next_target = int(next_state[0])

                # Q-learning update
                best_next_action = np.argmax(self.q_table[next_target])
                td_target = reward + self.discount_factor * self.q_table[next_target, best_next_action]
                td_error = td_target - self.q_table[current_target, action]
                self.q_table[current_target, action] += self.learning_rate * td_error

                current_target = next_target

            if episode % 100 == 0:
                print(f"Episode {episode} completed")

    def get_optimal_policy(self):
        return np.argmax(self.q_table, axis=1)

if __name__ == "__main__":
    num_targets = 100
    env = TSP(num_targets)
    solver = DPSolver(env)

    print("Training DP solver...")
    solver.train()

    optimal_policy = solver.get_optimal_policy()
    print("Optimal policy:", optimal_policy)

    # Evaluate the policy
    state, _ = env.reset()
    total_reward = 0
    for _ in range(100):
        action = optimal_policy[int(state[0])]
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"Total reward using optimal policy: {total_reward}")
