import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# parameters 
alpha = 0.1  
gamma = 0.95  
epsilon = 0.5 
epsilon_decay = 0.995  
epsilon_min = 0.01 
num_episodes = 10000 
max_steps = 10  

Q = defaultdict(lambda: np.zeros(env.action_space.n)) # Init Q-table as a defaultdict

def epsilon_greedy_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[state])  # Exploit

# Q-Learning 
def q_learning(env, num_episodes, max_steps, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
    episode_rewards = []  # total reward / episode
    
    for episode in range(num_episodes):
        # Reset the environment to get the initial state
        state, _ = env.reset()
        state = tuple(state)  # Convert state to tuple for Q-table indexing
        total_reward = 0
        
        for step in range(max_steps):
        
            action = epsilon_greedy_action(state, epsilon) # epsilon- greedy a selection
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_state) # Take action and observe the next state and reward
            
            # Update Q-value 
            best_next_action = np.argmax(Q[next_state])  # best a for s_n+1
            td_target = reward + gamma * Q[next_state][best_next_action]  # TD target
            td_error = td_target - Q[state][action] 
            Q[state][action] += alpha * td_error 
            
            total_reward += reward  
            state = next_state  
            
            if terminated or truncated:
                break
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min) #epsilon decay
        
        episode_rewards.append(total_reward)
        
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
    
    return episode_rewards

# Train the agent using Q-learning
env = ModTSP(num_targets=10)
episode_rewards = q_learning(env, num_episodes, max_steps, alpha, gamma, epsilon, epsilon_decay, epsilon_min)

# episode vs cumulative reward to show convergence
plt.plot(range(num_episodes), episode_rewards)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Episode vs Cumulative Reward (Q-Learning on Modified TSP)')
plt.ylim(-70000,400) 
plt.show()