import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# SARSA Hyperparameters
learning_rate = 0.1 
discount_factor = 0.95  
initial_epsilon = 0.5  
epsilon_decay_rate = 0.5  
min_epsilon = 0.01  
total_episodes = 100000  
max_steps_per_episode = 10  

# Initialize the Q-table 
Q_table = defaultdict(lambda: np.zeros(env.action_space.n))

def select_action(state, epsilon):
    """Returns an action using the ε-greedy policy."""
    if np.random.rand() < epsilon:  # Explore with ε
        return env.action_space.sample()
    else:  
        return np.argmax(Q_table[state])

# SARSA Algorithm
def sarsa(env, episodes, steps, alpha, gamma, epsilon, epsilon_decay, min_epsilon):
    episode_rewards = []  
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = tuple(state)  
        
        action = select_action(state, epsilon) # ε
        total_reward = 0  
        
        for step in range(steps):
            # Take the action and observe the next state, reward, and whether the episode is done
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_state)
            
            next_action = select_action(next_state, epsilon) # ε
            
            # Update Q-value 
            td_target = reward + gamma * Q_table[next_state][next_action]  
            td_error = td_target - Q_table[state][action]  
            Q_table[state][action] += alpha * td_error  
            
            total_reward += reward 
     
            state, action = next_state, next_action
            
            if terminated or truncated:
                break
        
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        
        episode_rewards.append(total_reward)
 
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
    
    return episode_rewards

env = ModTSP(num_targets=10)

# Train the agent using SARSA and get the rewards
episode_rewards = sarsa(
    env,
    episodes=total_episodes,
    steps=max_steps_per_episode,
    alpha=learning_rate,
    gamma=discount_factor,
    epsilon=initial_epsilon,
    epsilon_decay=epsilon_decay_rate,
    min_epsilon=min_epsilon
)

plt.plot(range(total_episodes), episode_rewards)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Episode vs Cumulative Reward (SARSA on Modified TSP)')
plt.ylim(-70000, 400)  
plt.grid(True) 
plt.show()