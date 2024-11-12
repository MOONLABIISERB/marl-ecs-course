
import numpy as np
import matplotlib.pyplot as plt
from Fixed_env import FixedMAPFEnvironment

class FixedPositionQLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1):
        self.q_table = np.zeros((state_space, state_space, action_space))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)
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
    Check for collisions between agents
    Returns True if collision detected, False otherwise
    """
    # Check direct swapping (agents crossing paths)
    for i in range(len(current_state)):
        for j in range(i+1, len(current_state)):
            # Check if agents swap positions
            if (current_state[i] == next_state[j]) and (current_state[j] == next_state[i]):
                return True
            
            # Check if agents move to the same cell
            if next_state[i] == next_state[j]:
                return True
    
    return False

def train_fixed_mapf(seed=42, episodes=5000, max_steps=1000):
    np.random.seed(seed)
    env = FixedMAPFEnvironment()
    
    # Initialize agents
    agents = [FixedPositionQLearningAgent(env.grid_size, 5) for _ in range(4)]
    
    # Tracking rewards
    episode_rewards = np.zeros((episodes, 4))
    cumulative_rewards = np.zeros((episodes, 4))
    collision_counts = np.zeros(episodes)
    
    for episode in range(episodes):
        current_state = env.initialize_scenario()
        episode_reward = np.zeros(4)
        collision_this_episode = False
        
        for step in range(max_steps):
            # Select actions for each agent
            actions = [agent.select_action(pos) for agent, pos in zip(agents, current_state)]
            
            # Execute actions and get next state
            next_state, rewards, done = env.execute_action(current_state, actions)
            
            # Check for collisions
            if check_collision(current_state, next_state):
                # Penalize agents for collision
                rewards = [reward - 10 for reward in rewards]
                collision_this_episode = True
            
            # Update Q-values for each agent
            for i in range(4):
                agents[i].update_q_value(current_state[i], actions[i], rewards[i], next_state[i])
                episode_reward[i] += rewards[i]
            
            current_state = next_state
            
            if done:
                break
        
        # Record collision count
        collision_counts[episode] = int(collision_this_episode)
        
        episode_rewards[episode] = episode_reward
        
        # Calculate cumulative average rewards
        if episode == 0:
            cumulative_rewards[episode] = episode_reward
        else:
            cumulative_rewards[episode] = cumulative_rewards[episode-1] + episode_reward
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Average Reward = {np.mean(episode_reward):.2f}, Collision = {collision_this_episode}")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Agent Rewards Plot
    for i in range(4):
        ax1.plot(episode_rewards[:, i], alpha=0.4, label=f'Agent {i} Episode Reward')
        ax1.plot(cumulative_rewards[:, i] / (np.arange(episodes) + 1), 
                 label=f'Agent {i} Cumulative Avg Reward')
    
    ax1.set_title('Fixed Position MAPF: Agent Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    
    # Collision Frequency Plot
    ax2.plot(np.cumsum(collision_counts) / (np.arange(episodes) + 1), color='red')
    ax2.set_title('Collision Frequency over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Collision Rate')
    
    plt.tight_layout()
    plt.savefig('fixed_mapf_rewards_and_collisions.png')
    plt.show()
    
    return agents, episode_rewards, collision_counts
