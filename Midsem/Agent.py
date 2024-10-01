import numpy as np
import gymnasium as gym
from collections import defaultdict
from modified_tsp import ModTSP
import matplotlib.pyplot as plt
import pandas as pd
import os


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """Initialize the Q-Learning agent."""
        self.env = env
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay after each episode
        self.epsilon_min = epsilon_min  # Minimum value of epsilon
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))# Q-table
        self.episode_rewards = []   # Rewards per episode
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore: random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: best action from Q-table
    
    def update_q_table(self, state, action, reward, next_state):
        """Update the Q-table using the Q-learning formula."""
        best_next_action = np.argmax(self.q_table[next_state])  # Best action for the next state
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
    def save_q_table(self, filename='q_table.csv'):
        """Save the Q-table to a file."""
        q_table_df = pd.DataFrame.from_dict(self.q_table, orient='index')
        q_table_df.to_csv(filename, header=False)  # Save as CSV without headers


    def load_q_table(self, filename='q_table.csv'):
    # """Load the Q-table from a CSV file."""
        if os.path.getsize(filename) == 0:  # Check if file is empty
          print("Q-table file is empty, starting with a fresh Q-table.")
          return  # Early return, Q-table remains as initialized

        q_table_df = pd.read_csv(filename, header=None, index_col=0)  # Load CSV
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n), q_table_df.to_dict(orient='index'))


    def save_episode_rewards(self, filename='episode_rewards.csv'):
        """Save the episode rewards to a CSV file."""
        rewards_df = pd.DataFrame(self.episode_rewards, columns=['Rewards'])
        rewards_df.to_csv(filename, index=False)  # Save rewards as CSV

    def load_episode_rewards(self, filename='episode_rewards.csv'):
        """Load the episode rewards from a CSV file."""
        if os.path.getsize(filename) == 0:  # Check if file is empty
         print("Episode rewards file is empty, starting with an empty rewards list.")
         return  # Early return, rewards remains empty

        rewards_df = pd.read_csv(filename)  # Load CSV
        self.episode_rewards = rewards_df['Rewards'].tolist()  # Convert back to listt

    
    def train(self,  num_episodes=1000, save_interval=100, q_table_filename='q_table.csv', rewards_filename='episode_rewards.csv'):
        """Train the Q-learning agent."""
        
         # Load Q-table if it exists
        try:
           self.load_q_table(q_table_filename)
           print("Loaded Q-table successfully.")
        except FileNotFoundError:
           print("No existing Q-table found, starting fresh.")
        
        # episode_rewards = []
        # Load episode rewards if it exists
        try:
            self.load_episode_rewards(rewards_filename)
            print("Loaded episode rewards successfully.")
        except FileNotFoundError:
            print("No existing episode rewards found, starting fresh.")
        

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            state = tuple(state)  # Convert state to tuple for Q-table indexing
            total_reward = 0

            for step in range(self.env.max_steps):
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = tuple(next_state)

                self.update_q_table(state, action, reward, next_state)

                state = next_state
                total_reward += reward

                if terminated or truncated:
                    break

            self.episode_rewards.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)  # Decay epsilon

            print(f"Episode {ep} - Total reward: {total_reward} - Epsilon: {self.epsilon}")
            # Save the Q-table at specified intervals
            if ep % save_interval == 0:  # Save every `save_interval` episodes
              self.save_q_table(q_table_filename)
              self.save_episode_rewards(rewards_filename)
              print(f"Saved Q-table after episode {ep}.")

         # Save the final Q-table after all episodes
        self.save_q_table(q_table_filename)
        self.save_episode_rewards(rewards_filename)
        print("Final Q-table saved.")

        return self.episode_rewards
    
def plot_rewards(rewards, window_size=50):
    """Plot original and smoothed rewards."""
    # Compute the moving average (smoothed rewards)
    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    # Plot the original rewards and smoothed rewards
    plt.figure(figsize=(10, 6))
    
    # Plot original rewards (in gray)
    plt.plot(rewards, color='gray', label='Original Rewards', alpha=0.3)
    
    # Plot smoothed rewards (in red)
    plt.plot(range(window_size - 1, len(rewards)), smoothed_rewards, color='red', label='Smoothed Rewards')
    
    # Adding labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Original vs Smoothed Rewards per Episode')

    # Adding grid and legend
    plt.grid(True)
    plt.legend()
    
    # Show the plot
    plt.show()


def main():
    env = ModTSP(num_targets=10)

    # Instantiate the Q-Learning agent
    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)
    
    # Specify the file names for Q-table and rewards
    q_table_filename = 'q_table.csv'
    rewards_filename = 'episode_rewards.csv'

    # Clear the saved files before starting training
    for filename in [q_table_filename, rewards_filename]:
        if os.path.exists(filename):
            open(filename, 'w').close()  # Empty the file

    # Train the agent
    rewards = agent.train(num_episodes=1000)
    plot_rewards(rewards, window_size=50)

if __name__ == "__main__":
    main()
