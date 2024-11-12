import numpy as np
import matplotlib.pyplot as plt
from Random_env import RandomMAPFEnvironment

class RandomPositionQLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1):
        self.q_table = np.zeros((state_space, state_space, action_space))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)  # Random action: 0-4 (4 actions + stay)
        return np.argmax(self.q_table[state[0], state[1]])

    def update_q_value(self, state, action, reward, next_state):
        curr_x, curr_y = state
        next_x, next_y = next_state
        current_q = self.q_table[curr_x, curr_y, action]
        max_next_q = np.max(self.q_table[next_x, next_y])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[curr_x, curr_y, action] = new_q

def check_collision(current_state, next_state):
    """
    Check for collisions between agents.
    Returns True if collision detected, False otherwise.
    """
    for i in range(len(current_state)):
        for j in range(i + 1, len(current_state)):
            # Check for direct swapping or moving to the same cell
            if (current_state[i] == next_state[j] and current_state[j] == next_state[i]) or \
               (next_state[i] == next_state[j]):
                return True
    return False

def train_random_mapf(seed=42, episodes=5000, max_steps=1000):
    np.random.seed(seed)
    env = RandomMAPFEnvironment()

    # Initialize agents
    agents = [RandomPositionQLearningAgent(env.grid_size, 5) for _ in range(env.num_agents)]

    # Tracking rewards and collisions
    episode_rewards = np.zeros((episodes, env.num_agents))
    cumulative_rewards = np.zeros((episodes, env.num_agents))
    collision_counts = np.zeros(episodes)

    for episode in range(episodes):
        # Initialize state: current_state should be a list of tuples (x, y) for each agent
        current_state = env.initialize_scenario()
        episode_reward = np.zeros(env.num_agents)
        collision_this_episode = False

        for step in range(max_steps):
            actions = [agent.select_action(pos) for agent, pos in zip(agents, current_state)]
            next_state, rewards, done = env.execute_action(current_state, actions)

            # Check if rewards is a single integer, if so distribute it across all agents
            if isinstance(rewards, int):  # Handle case where rewards is a single integer
                rewards = [rewards] * env.num_agents  # Assign the same reward to all agents

            # Ensure next_state is a list of tuples (x, y)
            if isinstance(next_state, list):
                assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in next_state), \
                    "Each agent's next state should be a tuple of coordinates (x, y)."

            # Update Q-values for each agent
            for i in range(env.num_agents):
                # Ensure state and next_state are properly indexed (i.e., they should be tuples)
                agents[i].update_q_value(current_state[i], actions[i], rewards[i], next_state[i])
                episode_reward[i] += rewards[i]

            current_state = next_state

            # Check for collisions
            if check_collision(current_state, next_state):
                collision_counts[episode] += 1
                collision_this_episode = True

            if done:
                break

        # Apply penalty (-1 for each step until all agents reach their goals)
        episode_rewards[episode] = episode_reward

        # Cumulative rewards for plotting
        if episode == 0:
            cumulative_rewards[episode] = episode_reward
        else:
            cumulative_rewards[episode] = cumulative_rewards[episode - 1] + episode_reward

        if episode % 100 == 0:
            print(f"Episode {episode}: Average Reward = {np.mean(episode_reward):.2f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Agent Rewards Plot
    for i in range(4):
        ax1.plot(episode_rewards[:, i], alpha=0.4, label=f'Agent {i} Episode Reward')
        ax1.plot(cumulative_rewards[:, i] / (np.arange(episodes) + 1),
                 label=f'Agent {i} Cumulative Avg Reward')

    ax1.set_title('Random Position MAPF: Agent Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()

    # Collision Frequency Plot
    ax2.plot(np.cumsum(collision_counts) / (np.arange(episodes) + 1), color='red')
    ax2.set_title('Collision Frequency over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Collision Rate')

    plt.tight_layout()
    plt.savefig('random_mapf_rewards_and_collisions.png')
    plt.show()

    return agents, episode_rewards, collision_counts

# Main execution block
if __name__ == "__main__":
    train_random_mapf()
