import os
import numpy as np
import pickle
from modified_tsp import ModTSP

# Q-Learning Agent for testing
class QLearningAgent:
    def __init__(self, num_states, num_actions, q_table_file):
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))
        self.load(q_table_file)

    def get_action(self, state, exploit=True):
        """Always choose the best action based on the Q-table (exploit mode)."""
        return np.argmax(self.q_table[state])

    def load(self, file_name):
        """Load the Q-table from a file."""
        with open(file_name, 'rb') as f:
            self.q_table = pickle.load(f)

# Testing the Q-learning agent
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
        print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")

    print(f"\nAverage Reward over {num_test_episodes} episodes: {np.mean(total_rewards)}")
    return total_rewards

# Main Execution for Testing
def main():
    num_targets = 10
    num_test_episodes = 10
    model_files = [
        "trained_q_table_Model_1.pkl",
        "trained_q_table_Model_2.pkl",
        "trained_q_table_Model_3.pkl"
        "trained_q_table_Model_4.pkl"
    ]

    # Load environment
    env = ModTSP(num_targets)

    # Check for available models and test on the first one found
    model_found = False
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"\nTesting on model: {model_file}")
            agent = QLearningAgent(num_states=num_targets, num_actions=num_targets, q_table_file=model_file)
            test_q_learning(env, agent, num_test_episodes)
            model_found = True
            break

    if not model_found:
        print("\nNo model found. Please ensure at least one model file is available.")

if __name__ == "__main__":
    main()
