import numpy as np
from numpy import typing as npt
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Import the environment and QLearner class from your main script

class QLearner:
    """Q-learning agent for testing in the Modified TSP environment."""

    def __init__(self, num_actions, q_matrix):
        """Initialize the Q-learning agent for testing with pre-learned Q-matrix."""
        self.num_actions = num_actions
        self.q_matrix = q_matrix  # Pre-trained Q-values

    def select_action(self, current_state):
        """Select the best action using a greedy policy based on the current state's Q-values."""
        state_key = str(current_state)
        if state_key in self.q_matrix:
            return np.argmax(self.q_matrix[state_key])  # Choose the best action based on Q-values
        else:
            return np.random.randint(self.num_actions)  # Random action if state is unseen


def test_agent(env, agent, num_episodes=100):
    """Test the agent in the environment without updating Q-values."""
    ep_rets = []  # Store cumulative rewards for each episode

    for ep in range(num_episodes):
        state, _ = env.reset()  # Reset the environment
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)  # Select action using the trained policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state  # Move to the next state

        ep_rets.append(total_reward)
        print(f"Test Episode {ep}: Total Reward Collected: {total_reward}")

    # Plot test results
    plt.plot(ep_rets, label='Test Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Test Cumulative Reward per Episode')
    plt.legend()
    plt.show()

    avg_reward = np.mean(ep_rets)
    print(f"Average Test Reward over {num_episodes} episodes: {avg_reward}")

    return ep_rets


if __name__ == "__main__":
    # Load pre-trained Q-values (replace 'q_matrix.npy' with your Q-value storage)
    try:
        q_matrix = np.load('q_matrix.npy', allow_pickle=True).item()
    except FileNotFoundError:
        print("No trained Q-matrix found! Please ensure you've trained and saved the Q-values.")
        exit()

    # Initialize the test environment
    num_targets = 5
    shuffle_time = 10
    env = ModTSP(num_targets, shuffle_time=shuffle_time)

    # Initialize the Q-learning agent with pre-trained Q-values
    agent = QLearner(num_actions=num_targets, q_matrix=q_matrix)

    # Test the agent
    test_agent(env, agent, num_episodes=100)
