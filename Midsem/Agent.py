import numpy as np
import gymnasium as gym
from collections import defaultdict
from modified_tsp import ModTSP
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """Initialize the Q-Learning agent."""
        self.env = env
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay after each episode
        self.epsilon_min = epsilon_min  # Minimum value of epsilon
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))  # Q-table
    
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
    
    def train(self, num_episodes=1000, max_steps_per_episode=100):
        """Train the Q-learning agent."""
        episode_rewards = []

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            state = tuple(state)  # Convert state to tuple for Q-table indexing
            total_reward = 0

            for step in range(max_steps_per_episode):
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = tuple(next_state)

                self.update_q_table(state, action, reward, next_state)

                state = next_state
                total_reward += reward

                if terminated or truncated:
                    break

            episode_rewards.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)  # Decay epsilon

            # print(f"Episode {ep} - Total reward: {total_reward} - Epsilon: {self.epsilon}")

        return episode_rewards
    
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

    # Train the agent
    rewards = agent.train(num_episodes=1000)
    plot_rewards(rewards, window_size=50)

    # Plot the rewards per episode
    # import matplotlib.pyplot as plt
    # plt.plot(rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title('Episode vs Total Reward')
    # plt.show()

if __name__ == "__main__":
    main()
