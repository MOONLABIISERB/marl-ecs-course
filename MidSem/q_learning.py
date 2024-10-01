import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from numpy import typing as npt
from modified_tsp import ModTSP

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.01, discount_factor=0.85, epsilon=1.0, epsilon_decay=0.9995):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.q_table = np.zeros((num_states, num_actions))

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        """Q-value update based on TD-learning."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# Training the Q-learning agent
def train_q_learning(env, agent, num_episodes):
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-100:])}")

    return episode_rewards


def plot_rewards(rewards):
    """Plot rewards with moving average."""
    window_size = 200
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, color="orange", label="Raw Rewards")
    plt.plot(range(window_size - 1, len(rewards)), moving_avg, color="blue", label=f"Moving Average (window={window_size})")
    plt.title("Episode Rewards over Time with Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("episode_rewards_plot.png")  # Save as PNG file
    plt.legend()
    plt.show()


# Main Execution
def main():
    num_targets = 10
    num_episodes = 10000

    env = ModTSP(num_targets)
    agent = QLearningAgent(num_states=num_targets, num_actions=num_targets)

    rewards = train_q_learning(env, agent, num_episodes)
    plot_rewards(rewards)
# Summary of results
    print("\nTraining Summary:")
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Maximum Reward Achieved: {np.max(rewards)}")
    print(f"Final Epsilon (Exploration Rate): {agent.epsilon:.4f}")

if __name__ == "__main__":
    main()
