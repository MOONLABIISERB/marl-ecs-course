#Shivendra Nath MIshra
# Roll Number 2320703
# MARL MIDSEM Code forsolving TSP environment using Q-learning


from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
from numpy import typing as npt
from modified_tsp import ModTSP 
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, discount_factor: float = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.q_table = {}

    def get_action(self, state: Tuple, epsilon: float) -> int:
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state_key])

    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple, done: bool):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        target = reward + (1 - done) * self.gamma * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])

    def _get_state_key(self, state: Tuple) -> Tuple:
        return tuple(state)

def get_epsilon(episode: int, min_epsilon: float = 0.01, max_epsilon: float = 0.7, decay_rate: float = 0.0005) -> float:
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

def smooth_data(data: List[float], window_size: int = 100) -> List[float]:
    """Apply moving average to smooth the data."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def plot_results(episodes, total_rewards, avg_rewards, avg_distances, epsilons, cumulative_rewards):
    window_size = 100  
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    fig.suptitle('Q-Learning Performance for Modified TSP (Smoothed)')

    smooth_episodes = episodes[window_size-1:]

    axs[0, 0].plot(smooth_episodes, smooth_data(total_rewards, window_size))
    axs[0, 0].set_title('Total Reward per Episode (Smoothed)')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')

    axs[0, 1].plot(smooth_episodes, smooth_data(avg_rewards, window_size))
    axs[0, 1].set_title('Average Reward per Episode (Smoothed)')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Average Reward')

    axs[1, 0].plot(smooth_episodes, smooth_data(avg_distances, window_size))
    axs[1, 0].set_title('Average Distance per Episode (Smoothed)')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Average Distance')

    axs[1, 1].plot(episodes, epsilons)  
    axs[1, 1].set_title('Epsilon Decay')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Epsilon')

    axs[2, 0].plot(smooth_episodes, smooth_data(cumulative_rewards, window_size))
    axs[2, 0].set_title('Cumulative Reward (Smoothed)')
    axs[2, 0].set_xlabel('Episode')
    axs[2, 0].set_ylabel('Cumulative Reward')

    plt.tight_layout()
    plt.show()

def main():
    num_targets = 10
    num_episodes = 100000
    max_steps_per_episode = num_targets

    env = ModTSP(num_targets)
    agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n)

    
    episodes = []
    total_rewards = []
    avg_rewards = []
    avg_distances = []
    epsilons = []
    cumulative_reward = 0
    cumulative_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        epsilon = get_epsilon(episode)

        episode_rewards = []
        episode_distances = []

        for step in range(max_steps_per_episode):
            action = agent.get_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            episode_rewards.append(reward)
            if 'distance_travelled' in info:
                episode_distances.append(info["distance_travelled"])

            if done:
                break

        avg_reward = np.mean(episode_rewards)
        avg_distance = np.mean(episode_distances) if episode_distances else 0

        cumulative_reward += total_reward

        # Storing of results metrics for plotting
        episodes.append(episode)
        total_rewards.append(total_reward)
        avg_rewards.append(avg_reward)
        avg_distances.append(avg_distance)
        epsilons.append(epsilon)
        cumulative_rewards.append(cumulative_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Avg Reward: {avg_reward:.2f}, Avg Distance: {avg_distance:.2f}, "
                  f"Epsilon: {epsilon:.2f}, Cumulative Reward: {cumulative_reward:.2f}")

    # Ploting results
    plot_results(episodes, total_rewards, avg_rewards, avg_distances, epsilons, cumulative_rewards)

if __name__ == "__main__":
    main()