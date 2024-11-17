import time
from mapf_env import PressurePlate
import numpy as np
import matplotlib.pyplot as plt

# Initialize the environment
env = PressurePlate(10, 10)
obs_ = env.reset()

# Parameters
n_agents = 4           # Number of agents
n_actions = 5          # Number of possible actions (0: Left, 1: Right, 2: Up, 3: Down, 4: No-op)
grid_size = (10, 10)   # Grid dimensions
max_episodes = 500   # Maximum number of episodes
max_steps = 200        # Maximum steps per episode

# Hyperparameters
alpha = 0.1            # Learning rate
gamma = 0.99           # Discount factor
epsilon = 0.1          # Exploration probability

# Initialize policy and Q-values
policy = np.zeros((n_agents, grid_size[1], grid_size[0]), dtype=int)
Q = np.zeros((n_agents, grid_size[1], grid_size[0], n_actions), dtype=float)

# Initialize total rewards
total_reward = [float(0)] * n_agents
avg_reward = []

episode_reward = []

# Evaluate the current policy to select actions for all agents
def policy_eval(state):
    actions = []
    for i in range(n_agents):
        x, y = state[i][0], state[i][1]
        actions.append(policy[i, x, y])
    return actions

# Simulate state transition based on action, with collision detection
def state_transition(state, action):
    stat = state.copy()
    proposed_pos = [state[0], state[1]]
    
    if action == 0:  # Move Left
        proposed_pos[1] -= 1
    elif action == 1:  # Move Right
        proposed_pos[1] += 1
    elif action == 2:  # Move Up
        proposed_pos[0] -= 1
    elif action == 3:  # Move Down
        proposed_pos[0] += 1
    # action == 4 is NOOP (no operation)

    # Update state only if there's no collision
    if not env._detect_collision(proposed_pos):
        stat = proposed_pos
    
    return stat

# Calculate the reward for a given agent based on its state
def agent_reward(next_state, agent):
    goal_loc = (env.plates[agent].x, env.plates[agent].y)  # Goal location of the agent
    agent_loc = (next_state[agent][0], next_state[agent][1])  # Current location of the agent
    
    # Reward is 0 if the agent is at its goal, otherwise -1
    return 0 if agent_loc == goal_loc else -1

# Update Q-values and select the best action for a given agent
def q_value(Q, state, agent):
    for a in range(n_actions):
        st = state.copy()
        st_ = state.copy()
        next_st = state_transition(st[agent], a)
        st_[agent] = next_st       
        reward = agent_reward(st_, agent)
        
        # Update Q-value using the Bellman equation
        Q[st[agent][0], st[agent][1], a] += alpha * (
            reward + gamma * np.max(Q[next_st[0], next_st[1]]) - Q[st[agent][0], st[agent][1], a]
        )
    
    # Return the best action (highest Q-value)
    return np.argmax(Q[state[agent][0], state[agent][1]])


def plot(avg_reward, ep_reward):
    ep_rd = np.array(ep_reward)
    avg_rd = np.array(avg_reward)
    length = np.arange(1, max_episodes+1)
    
    plt.figure(figsize=(12, 6))
        
    plt.plot(length, ep_rd[:,0], color='#82d6f3',  alpha=0.4 )
    plt.plot(length, ep_rd[:,1], color='#82f397',  alpha=0.4 )
    plt.plot(length, ep_rd[:,2], color='#d782f3',  alpha=0.4 )
    plt.plot(length, ep_rd[:,3], color='#ee6f88', alpha=0.4 )

    plt.plot(length, avg_rd[:,0], color='#3099ec', label='Cumulative Reward for agent_0' )
    plt.plot(length, avg_rd[:,1], color='#22cf41', label='Cumulative Reward for agent_1' )
    plt.plot(length, avg_rd[:,2], color='#a31cbb', label='Cumulative Reward for agent_2' )
    plt.plot(length, avg_rd[:,3], color='#e71313', label='Cumulative Reward for agent_3' )
    
    plt.title('Episode vs Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    
    plt.legend()
    
    plt.savefig("output.jpg")
    
    plt.show()
    
    

# Main function to train and evaluate the agents
def main():
    for episode in range(max_episodes):
        obs_ = env.reset()  # Reset the environment
        state = np.array([obs_[i][3:5] for i in range(n_agents)], dtype=int)  # Initial states of agents
        ep_reward = [float(0)] * n_agents  # Track rewards for this episode
        rd = [float(0)] * n_agents # Track reward for this episode
        for step in range(max_steps):
            # Roll out algorithm for each agent
            for agent in range(n_agents): 
                action = np.array([4] * n_agents, dtype=int)  # Default actions (No-op)
                act = q_value(Q[agent], state, agent)  # Select action using Q-learning
                action[agent] = act  # Update action for the current agent
                
                # Step in the environment
                obs, reward, done, _ = env.step(action)
                policy[agent, state[agent][0], state[agent][1]] = act  # Update the policy
                
                # Update state and rewards
                state[agent] = obs[agent][3:5]
                ep_reward[agent] += reward[agent]
                
                # Render the environment
                # if episode >= 490:
                #     env.render()   
                #     time.sleep(0.1)
                
                # Break if all agents have reached their goals
                if sum(reward) == 0:
                    break
        
        # Update total and average rewards
        for i in range(n_agents):
            total_reward[i] += ep_reward[i]
            rd[i] = total_reward[i]/(episode + 1)
        avg_reward.append(rd)

        episode_reward.append(ep_reward)
       
    # To plot the graph between cummulative reward and episode reward    
    plot(avg_reward, episode_reward)
    
    # Prints the total time taken by the agents to reach their goals
    print(f" Total time for all agents to reach the goal {-1 * min(ep_reward)}")   

# Run the main function
if __name__ == "__main__":
    main()

