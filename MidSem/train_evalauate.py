import numpy as np
import matplotlib.pyplot as plt
import pickle
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

    def get_action(self, state, exploit=False):
        """Epsilon-greedy action selection with option to exploit only."""
        if exploit:
            # Exploit only: return the action with the highest Q-value
            return np.argmax(self.q_table[state])
        else:
            # Epsilon-greedy strategy: either explore or exploit
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

    def save(self, file_name):
        """Save the Q-table to a file."""
        with open(file_name, 'wb') as f:
            pickle.dump(self.q_table, f)

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

    return episode_rewards

# Function to test the agent
def test_q_learning(env, agent, num_test_episodes=10):
    """Run the trained agent and report the rewards."""
    total_rewards = []
    for episode in range(num_test_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, exploit=True)  # Exploit only (no exploration)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)  # Return average reward

# Plot training rewards for all models on a single plot
def plot_all_training_rewards(all_rewards, best_model_idx):
    """Plot training rewards for all models on a single plot."""
    plt.figure(figsize=(12, 8))
    
    window_size = 200
    for idx, (model_name, rewards) in enumerate(all_rewards):
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
        if idx == best_model_idx:
            plt.plot(range(window_size - 1, len(rewards)), moving_avg, label=f"{model_name} (Best)", linewidth=3, color='red')
        else:
            plt.plot(range(window_size - 1, len(rewards)), moving_avg, label=model_name, alpha=0.5)

    plt.title("Training Rewards for All Models (Best Model Highlighted)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("all_models_training_rewards.png")
    plt.show()

# Plot comparison of test performances
def plot_test_comparison(models, test_rewards, best_model_idx):
    """Plot the test reward comparison between different models."""
    plt.figure(figsize=(12, 8))
    bar_colors = ['orange' if i != best_model_idx else 'red' for i in range(len(models))]
    plt.bar(models, test_rewards, color=bar_colors)
    plt.title("Test Reward Comparison between Models (Best Model Highlighted)")
    plt.xlabel("Model")
    plt.ylabel("Average Test Reward")
    plt.grid(True)
    plt.savefig(f"test_reward_comparison.png")
    plt.show()

# Main function to train and evaluate with different hyperparameters
def main():
    num_targets = 10
    num_episodes = 10000
    num_test_episodes = 10

    # Define hyperparameter sets to explore
    hyperparameter_sets = [
        {'learning_rate': 0.01, 'discount_factor': 0.85, 'epsilon_decay': 0.9995},
        {'learning_rate': 0.01, 'discount_factor': 0.9,  'epsilon_decay': 0.999},
        {'learning_rate': 0.05, 'discount_factor': 0.85, 'epsilon_decay': 0.999},
        {'learning_rate': 0.1,  'discount_factor': 0.9,  'epsilon_decay': 0.995},
        # Add more hyperparameter sets if necessary
    ]

    best_model = None
    best_performance = float('-inf')
    best_hyperparams = None
    best_model_idx = -1

    all_models = []
    all_rewards = []
    all_test_rewards = []

    # Loop through each hyperparameter set
    for idx, hyperparams in enumerate(hyperparameter_sets):
        model_name = f"Model_{idx+1}"

        # Create environment and agent with current hyperparameters
        env = ModTSP(num_targets)
        agent = QLearningAgent(
            num_states=num_targets,
            num_actions=num_targets,
            learning_rate=hyperparams['learning_rate'],
            discount_factor=hyperparams['discount_factor'],
            epsilon_decay=hyperparams['epsilon_decay']
        )

        # Train the agent
        print(f"Training {model_name} with hyperparams: {hyperparams}")
        rewards = train_q_learning(env, agent, num_episodes)

        # Store the model's training rewards for plotting later
        all_rewards.append((model_name, rewards))

        # Test the agent and get performance
        avg_test_reward = test_q_learning(env, agent, num_test_episodes)
        print(f"{model_name} average test reward: {avg_test_reward}\n")

        # Store the model and its test reward for comparison later
        all_models.append(model_name)
        all_test_rewards.append(avg_test_reward)

        # Check if this model is the best so far
        if avg_test_reward > best_performance:
            best_performance = avg_test_reward
            best_model = model_name
            best_hyperparams = hyperparams
            best_model_idx = idx

    # Plot all training rewards, highlighting the best model
    plot_all_training_rewards(all_rewards, best_model_idx)

    # Plot test reward comparison, highlighting the best model
    plot_test_comparison(all_models, all_test_rewards, best_model_idx)

    # Save the best model
    env = ModTSP(num_targets)
    agent = QLearningAgent(
        num_states=num_targets,
        num_actions=num_targets,
        learning_rate=best_hyperparams['learning_rate'],
        discount_factor=best_hyperparams['discount_factor'],
        epsilon_decay=best_hyperparams['epsilon_decay']
    )
    agent.save(f"trained_q_table_{best_model}.pkl")

    print(f"\nBest performing model: {best_model}")
    print(f"Best performance: {best_performance}")
    print(f"Best hyperparameters: {best_hyperparams}")

if __name__ == "__main__":
    main()




